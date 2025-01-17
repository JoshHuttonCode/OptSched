//===- OptimizingScheduler.h - Combinatorial scheduler ----------*- C++ -*-===//
//
// Integrates an alternative scheduler into LLVM which implements a
// combinatorial scheduling algorithm.
//
//===----------------------------------------------------------------------===//

// #ifndef LLVM_OPT_SCHED_OPTIMIZING_SCHEDULER_H
// #define LLVM_OPT_SCHED_OPTIMIZING_SCHEDULER_H

#pragma once

#include "GCNSubtarget.h"
#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Debug.h"
#include "GCNRegPressure.h"
#include <chrono>
#include <memory>
#include <vector>

using namespace llvm;

namespace llvm {
namespace opt_sched {

class OptSchedDDGWrapperBasic;

class ILPMetrics {
  unsigned ScheduleLength;
  unsigned BubbleCycles;

public:
  ILPMetrics() = default;
  ILPMetrics(unsigned L, unsigned BC)
      : ScheduleLength(L), BubbleCycles(BC) {}
  unsigned getLength() const { return ScheduleLength; }
  unsigned getBubbles() const { return BubbleCycles; }
  unsigned getMetric() const {
    unsigned Metric = (BubbleCycles * ScaleFactor) / ScheduleLength;
    // Metric is zero if the amount of bubbles is less than 1% which is too
    // small. So, return 1.
    return Metric ? Metric : 1;
  }
  static const unsigned ScaleFactor;
};

inline raw_ostream &operator<<(raw_ostream &OS, const ILPMetrics &Sm) {
  dbgs() << "\n Schedule Metric (scaled by "
         << ILPMetrics::ScaleFactor
         << " ) is: " << Sm.getMetric() << " [ " << Sm.getBubbles() << "/"
         << Sm.getLength() << " ]\n";
  return OS;
}

// Evaluate schedules with AMDGPU target.
class ScheduleEvaluator {
private:
  ScheduleDAGOptSched &DAG;

  // RP before scheduling.
  GCNRegPressure RPBefore;

  // RP after scheduling.
  GCNRegPressure RPAfter;

  // ILP before scheduling.
  int64_t ILPBefore;

  // ILP after scheduling.
  int64_t ILPAfter;

  // Record the original instruction order.
  std::vector<MachineInstr *> Unsched;

  ILPMetrics calculateILP() const;

public:
  ScheduleEvaluator(ScheduleDAGOptSched &DAG) : DAG(DAG) {}

  virtual ~ScheduleEvaluator() = default;

  void calculateRPBefore();

  void calculateRPAfter();

  void calcualteILPBefore();

  void claculateILPAfter();

  // Returns RPAfter - RPBefore.
  unsigned getOccDifference() const;

  // Returns ILPAfter - ILPBefore.
  int64_t getILPDifference() const;

  unsigned getOccupancyBefore() const;

  void recordSchedule();

  void revertScheduling();
};

class ScheduleDAGOptSched : public ScheduleDAGMILive {
friend class ScheduleEvaluator;

private:
  enum SchedPassStrategy { OptSchedMinRP, OptSchedBalanced };

  // Vector of scheduling passes to execute.
  SmallVector<SchedPassStrategy, 4> SchedPasses;

protected:
  // Vector of regions recorded for later rescheduling
  SmallVector<
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>, 32>
      Regions;

  // Holds size of largest region. Used for device DDG allocation
  InstCount maxRegionSize;

  // Holds amount of edges in region with most edges
  // used for maxDDG allocation
  InstCount maxEdgeCnt;

  // Holds device pointer to device max DDG pointer
  DataDepGraph **dev_maxDDG;

  // Path to opt-sched config options directory.
  SmallString<128> PathCfg;

  // Path to the scheduler options configuration file for opt-sched.
  SmallString<128> PathCfgS;

  // Path to the list of hot functions to schedule using opt-sched.
  SmallString<128> PathCfgHF;

  // Path to the machine model specification file for opt-sched.
  SmallString<128> PathCfgMM;

  SmallString<128> PathCfgOCL;

  // Bool value indicating that the scheduler is in the second
  // pass. Used for the two pass scheduling approach.
  bool SecondPass;

  // Region number uniquely identifies DAGs.
  unsigned RegionNumber = ~0u;

  MachineSchedContext *C;

  // The OptSched target machine.
  std::unique_ptr<OptSchedTarget> OST;

  // into the OptSched machine model
  std::unique_ptr<OptSchedMachineModel> MM;

  // OptSched MachineModel on device
  MachineModel *dev_MM;

  // A list of functions that are indicated as candidates for the
  // OptScheduler
  Config HotFunctions;

  // A list of kernels / functions and the occupancy limit the maximizes performance
  Config OccupancyLimits;

  // Struct for setting the pruning strategy
  Pruning PruningStrategy;

  // If we should schedule for register pressure only and ignore ilp.
  bool SchedForRPOnly;

  // Flag indicating whether the optScheduler should be enabled for this
  // function
  bool OptSchedEnabled;

  // Flag indicating whether the two pass scheduling approach should be used
  // instead of the original one pass scheduling.
  bool TwoPassEnabled;

  // Flag indicating whether or not the two pass scheduling approach
  // has started. The two pass scheduling approach starts in finalizeSchedule.
  bool TwoPassSchedulingStarted;

  // Precision of latency info
  LATENCY_PRECISION LatencyPrecision;

  // A time limit for the whole region (basic block) in milliseconds.
  // Defaults to no limit.
  int RegionTimeout;

  // A time limit for each schedule length in milliseconds.
  int LengthTimeout;

  // Time limits for two pass scheduling. Same as the above time limits
  // but for each individual pass.
  int FirstPassRegionTimeout;
  int FirstPassLengthTimeout;
  int SecondPassRegionTimeout;
  int SecondPassLengthTimeout;

  int OccupancyLimit;

  bool ShouldLimitOccupancy;
  OCC_LIMIT_TYPE OccupancyLimitSource;

  bool DeviceACOEnabled;

  // How to interpret the timeout value? Timeout per instruction or
  // timout per block
  bool IsTimeoutPerInst;

  // The maximum number of instructions that a block can contain to be
  // Treat data dependencies of type ORDER as data dependencies
  bool TreatOrderAsDataDeps;

  // The number of bits in the hash table used in history-based domination.
  int16_t HistTableHashBits;

  // Whether to verify that calculated schedules are optimal. Defaults to NO.
  bool VerifySchedule;

  // Whether to enumerate schedules containing stalls (no-op instructions).
  // In certain cases, such as having unpipelined instructions, this may
  // result in a better schedule. Defaults to YES
  bool EnumStalls;

  // Whether to apply LLVM mutations to the DAG before scheduling
  bool EnableMutations;

  // The weight of the spill cost in the objective function. This factor
  // defines the importance of spill cost relative to schedule length. A good
  // value for this factor should be found experimentally, but is is expected
  // to be large on architectures with hardware scheduling like x86 (thus
  // making spill cost minimization the primary objective) and smaller on
  // architectures with in-order execution like SPARC (thus making scheduling
  // the primary objective).
  int SCW;

  // Evaluate scheduling regions and decide whether to revert to a previous
  // schedule or keep a new one.
  std::vector<ScheduleEvaluator> SchedEvals;

  // In ISO mode this is the original DAG before ISO conversion.
  std::vector<SUnit> OriginalDAG;

  // The schedule generated by LLVM for ISO mode.
  std::vector<unsigned> ISOSchedule;

  // The spill cost function to be used.
  SPILL_COST_FUNCTION SCF;

  // The algorithm to use for determining the lower bound. Valid values are
  LB_ALG LowerBoundAlgorithm;

  // The heuristic used for the list scheduler.
  SchedPriorities HeuristicPriorities;

  // The heuristic used for the enumerator.
  SchedPriorities EnumPriorities;

  // The heuristic used for the second pass enumerator in the two-pass
  // scheduling approach.
  SchedPriorities SecondPassEnumPriorities;

  // The heuristic used for the first portion of blocks in ACO.
  SchedPriorities AcoPriorities1;

  // The heuristic used for the second portion of blocks in ACO.
  SchedPriorities AcoPriorities2;

  // The heuristic used for the first portion of blocks in ACO
  // in the two-pass scheduling approach.
  SchedPriorities SecondPassAcoPriorities1;

  // The heuristic used for the second portion of blocks in ACO
  // in the two-pass scheduling approach.
  SchedPriorities SecondPassAcoPriorities2;

  // Static node superiority RP only graph transformation.
  bool StaticNodeSup;

  // Run multiple passes of the static node superiority algorithm
  // (StaticNodeSup must be enabled).
  bool MultiPassStaticNodeSup;

  // Should we run the LLVM converging scheduler as input to the enumerator.
  bool UseLLVMScheduler;

  // The number of simulated register spills in this function
  int SimulatedSpills;

  // What list scheduler should be used to find an initial feasible schedule.
  SchedulerType HeurSchedType;

  // Return the real register pressure for the current region. Only work for
  // AMDGPU target.
  GCNRegPressure getRealRegPressure() const;

  // Load config files for the OptScheduler and set flags
  void loadOptSchedConfig();

  // Get lower bound algorithm
  LB_ALG parseLowerBoundAlgorithm() const;

  // Get spill cost function
  SPILL_COST_FUNCTION parseSpillCostFunc() const;

  OCC_LIMIT_TYPE parseOccLimit(const std::string Str);

  // Return true if the OptScheduler should be enabled for the function this
  // ScheduleDAG was created for
  bool isOptSchedEnabled() const;

  // Return true if the two pass scheduling approach should be enabled
  bool isTwoPassEnabled() const;

  // get latency precision setting
  LATENCY_PRECISION fetchLatencyPrecision() const;

  // Get OptSched heuristic setting
  SchedPriorities parseHeuristic(const std::string &str);

  // Return true if we should print spill count for the current function
  bool shouldPrintSpills() const;

  // Reset the flags (e.g undef) before reverting scheduling
  void ResetFlags(SUnit &SU);
  
  // Add node to llvm schedule
  void ScheduleNode(SUnit *SU, unsigned CurCycle);

  // Setup dag and calculate register pressue in region
  void SetupLLVMDag();

  // Check for a mismatch between LLVM and OptSched register pressure values.
  bool rpMismatch(InstSchedule *sched);

  // Is simulated register allocation enabled.
  bool isSimRegAllocEnabled() const;

  // Find the real paths to optsched-cfg files.
  void getRealCfgPaths();

  // Create and add OptSched DDG mutations.
  void addGraphTransformations(OptSchedDDGWrapperBasic *BDDG);

public:
  ScheduleDAGOptSched(MachineSchedContext *C,
                      std::unique_ptr<MachineSchedStrategy> S);

  ~ScheduleDAGOptSched();

  // The fallback LLVM scheduler
  void fallbackScheduler();

  // Print out total block spills for the function.
  void finalizeSchedule() override;

  // Schedule the current region using the OptScheduler
  void schedule() override;

  // Setup and select schedulers for the two pass scheduling approach.
  virtual void initSchedulers();

  // Execute a scheduling pass on the function.
  void runSchedPass(SchedPassStrategy S);

  // Run OptSched in RP only configuration.
  void scheduleOptSchedMinRP();

  // Run OptSched in ILP/RP balanced mode.
  virtual void scheduleOptSchedBalanced();

  // Print info for all LLVM registers that are used or defined in the region.
  void dumpLLVMRegisters() const;

  // Getter for region number
  int getRegionNum() const { return RegionNumber; }

  // Return the boundary instruction for this region if it is not a sentinel
  // value.
  const MachineInstr *getRegionEnd() const {
    return (RegionEnd == BB->end() ? nullptr : &*RegionEnd);
  }

  LATENCY_PRECISION getLatencyType() const { return LatencyPrecision; }
};

} // namespace opt_sched
} // namespace llvm

// #endif // LLVM_OPT_SCHED_OPTIMIZING_SCHEDULER_H
