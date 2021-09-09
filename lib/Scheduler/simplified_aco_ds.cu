#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/simplified_aco_ds.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include "opt-sched/Scheduler/machine_model.h"
#include <algorithm>
#include <cstddef>
#include <utility>
//aco simplified ds impl

using namespace llvm::opt_sched;

//use the log message macro to make GPU porting easier
#define LOG_MESSAGE(...) Logger::Info(__VA_ARGS__)

// ----
// ACOReadyList
// ----

ACOReadyList::ACOReadyList() {
  InstrCount = 0;
  CurrentSize = 0;
  CurrentCapacity = PrimaryBufferCapacity = 0;
  Overflowed = false;

  // create new allocations for the data
  IntAllocation = nullptr;
  HeurAllocation = nullptr;
  ScoreAllocation = nullptr;

  //build shortcut pointers
  InstrBase = nullptr;
  ReadyOnBase = nullptr;
  HeurBase = nullptr;
  ScoreBase = nullptr;

}

ACOReadyList::ACOReadyList(InstCount RegionSize) {
  InstrCount = RegionSize;
  CurrentSize = 0;
  CurrentCapacity = PrimaryBufferCapacity = computePrimaryCapacity(InstrCount);
  Overflowed = false;

  // create new allocations for the data
  IntAllocation = new InstCount[2*CurrentCapacity];
  HeurAllocation = new HeurType[CurrentCapacity];
  ScoreAllocation = new pheromone_t[CurrentCapacity];

  //build shortcut pointers
  InstrBase = IntAllocation;
  ReadyOnBase = IntAllocation + CurrentCapacity;
  HeurBase = HeurAllocation;
  ScoreBase = ScoreAllocation;
}

ACOReadyList::ACOReadyList(const ACOReadyList &Other) {
  InstrCount = Other.InstrCount;
  PrimaryBufferCapacity = Other.PrimaryBufferCapacity;
  Overflowed = Other.Overflowed;
  CurrentCapacity = Other.CurrentCapacity;
  CurrentSize = Other.CurrentSize;

  // create new allocations for the data
  IntAllocation = new InstCount[2*CurrentCapacity];
  HeurAllocation = new HeurType[CurrentCapacity];
  ScoreAllocation = new pheromone_t[CurrentCapacity];

  //build shortcut pointers
  InstrBase = IntAllocation;
  ReadyOnBase = IntAllocation + CurrentCapacity;
  HeurBase = HeurAllocation;
  ScoreBase = ScoreAllocation;

  // copy the allocation's entries
  for (InstCount I = 0; I < CurrentSize; ++I) {
    InstrBase[I] = Other.InstrBase[I];
    ReadyOnBase[I] = Other.ReadyOnBase[I];
    HeurBase[I] = Other.HeurBase[I];
    ScoreBase[I] = Other.ScoreBase[I];
  }
}

ACOReadyList &ACOReadyList::operator=(const ACOReadyList &Other) {
  InstrCount = Other.InstrCount;
  PrimaryBufferCapacity = Other.PrimaryBufferCapacity;
  Overflowed = Other.Overflowed;
  CurrentCapacity = Other.CurrentCapacity;
  CurrentSize = Other.CurrentSize;

  // delete current allocations
  delete[] IntAllocation;
  delete[] HeurAllocation;
  delete[] ScoreAllocation;

  // create new allocations for the data
  IntAllocation = new InstCount[2*CurrentCapacity];
  HeurAllocation = new HeurType[CurrentCapacity];
  ScoreAllocation = new pheromone_t[CurrentCapacity];

  //build shortcut pointers
  InstrBase = IntAllocation;
  ReadyOnBase = IntAllocation + CurrentCapacity;
  HeurBase = HeurAllocation;
  ScoreBase = ScoreAllocation;

  // copy over the allocation's entries
  for (InstCount I = 0; I < CurrentSize; ++I) {
    InstrBase[I] = Other.InstrBase[I];
    ReadyOnBase[I] = Other.ReadyOnBase[I];
    HeurBase[I] = Other.HeurBase[I];
    ScoreBase[I] = Other.ScoreBase[I];
  }

  return *this;
}

ACOReadyList::ACOReadyList(ACOReadyList &&Other) noexcept {
  InstrCount = Other.InstrCount;
  PrimaryBufferCapacity = Other.PrimaryBufferCapacity;
  Overflowed = Other.Overflowed;
  CurrentCapacity = Other.CurrentCapacity;
  CurrentSize = Other.CurrentSize;

  // copy over the old ready lists allocations and set them to NULL
  // so that the data we took won't get deleted
  IntAllocation = Other.IntAllocation;
  HeurAllocation = Other.HeurAllocation;
  ScoreAllocation = Other.ScoreAllocation;
  Other.IntAllocation = nullptr;
  Other.HeurAllocation = nullptr;
  Other.ScoreAllocation = nullptr;

  InstrBase = Other.InstrBase;
  ReadyOnBase = Other.ReadyOnBase;
  HeurBase = Other.HeurBase;
  ScoreBase = Other.ScoreBase;
}

ACOReadyList &ACOReadyList::operator=(ACOReadyList &&Other) noexcept {
  InstrCount = Other.InstrCount;
  PrimaryBufferCapacity = Other.PrimaryBufferCapacity;
  Overflowed = Other.Overflowed;
  CurrentCapacity = Other.CurrentCapacity;
  CurrentSize = Other.CurrentSize;

  // swap the allocations to give Other our allocations to delete
  std::swap(IntAllocation, Other.IntAllocation);
  std::swap(HeurAllocation, Other.HeurAllocation);
  std::swap(ScoreAllocation, Other.ScoreAllocation);

  InstrBase = Other.InstrBase;
  ReadyOnBase = Other.ReadyOnBase;
  HeurBase = Other.HeurBase;
  ScoreBase = Other.ScoreBase;

  return *this;
}

ACOReadyList::~ACOReadyList() {
  delete[] IntAllocation;
  delete[] HeurAllocation;
  delete[] ScoreAllocation;
}


// This is just a heuristic for the ready list size.
// A better function should be chosen experimentally
InstCount ACOReadyList::computePrimaryCapacity(InstCount RegionSize) {
  //return std::max(32, RegionSize/4);
  return RegionSize;
}

__host__ __device__
void ACOReadyList::addInstructionToReadyList(const ACOReadyListEntry &Entry) {
  #ifdef __CUDA_ARCH__
    if (CurrentSize == CurrentCapacity) {
      printf("Ready List ran out of capacity and needs to be resized");
      exit(1);
    } else {
      //add the instruction to the ready list
      InstrBase[CurrentSize*numThreads_ + GLOBALTID] = Entry.InstId;
      ReadyOnBase[CurrentSize*numThreads_ + GLOBALTID] = Entry.ReadyOn;
      HeurBase[CurrentSize*numThreads_ + GLOBALTID] = Entry.Heuristic;
      ScoreBase[CurrentSize*numThreads_ + GLOBALTID] = Entry.Score;
      ++CurrentSize;
    }

    /*if (CurrentSize == CurrentCapacity) {
      int OldCap = CurrentCapacity;
      bool PrevOverflowed = Overflowed;

      // get a new allocation to put the data in
      // The expansion formula is to make the new allocation 1.5 times the size of the old one
      // consider making this formula more aggressive
      int NewCap = (OldCap + OldCap/2 + 1) * numThreads;
      InstCount *NewIntFallback = new InstCount[2*NewCap];
      HeurType *NewHeurFallback = new HeurType[NewCap];
      pheromone_t *NewScoreFallback = new pheromone_t[NewCap];

      // copy the data
      InstCount NewInstrOffset = 0, NewReadyOnOffset = NewCap, HeurOffset = 0, ScoreOffset = 0;
      for (int I = 0; I < CurrentSize; ++I) {
        NewIntFallback[numThreads*(NewInstrOffset + I) + GLOBALTID] = InstrBase[numThreads*I + GLOBALTID];
        NewIntFallback[numThreads*(NewReadyOnOffset + I) + GLOBALTID] = ReadyOnBase[numThreads*I + GLOBALTID];
        NewHeurFallback[numThreads*(HeurOffset + I) + GLOBALTID] = HeurBase[numThreads*I + GLOBALTID];
        NewScoreFallback[numThreads*(ScoreOffset + I) + GLOBALTID] = ScoreBase[numThreads*I + GLOBALTID];
    }*/
  #else
    // check to see if we need to expand the allocation/get a new allocation
    if (CurrentSize == CurrentCapacity) {
      int OldCap = CurrentCapacity;
      bool PrevOverflowed = Overflowed;

      // get a new allocation to put the data in
      // The expansion formula is to make the new allocation 1.5 times the size of the old one
      // consider making this formula more aggressive
      int NewCap = OldCap + OldCap/2 + 1;
      InstCount *NewIntFallback = new InstCount[2*NewCap];
      HeurType *NewHeurFallback = new HeurType[NewCap];
      pheromone_t *NewScoreFallback = new pheromone_t[NewCap];

      // copy the data
      InstCount NewInstrOffset = 0, NewReadyOnOffset = NewCap, HeurOffset = 0, ScoreOffset = 0;
      for (int I = 0; I < CurrentSize; ++I) {
        NewIntFallback[NewInstrOffset + I] = InstrBase[I];
        NewIntFallback[NewReadyOnOffset + I] = ReadyOnBase[I];
        NewHeurFallback[HeurOffset + I] = HeurBase[I];
        NewScoreFallback[ScoreOffset + I] = ScoreBase[I];
      }

      //delete the old allocations
      delete[] IntAllocation;
      delete[] HeurAllocation;
      delete[] ScoreAllocation;

      //copy the new allocations
      IntAllocation = NewIntFallback;
      HeurAllocation = NewHeurFallback;
      ScoreAllocation = NewScoreFallback;

      // update/recompute pointers and other values
      InstrBase = IntAllocation + NewInstrOffset;
      ReadyOnBase = IntAllocation + NewReadyOnOffset;
      HeurBase = HeurAllocation + HeurOffset;
      ScoreBase = ScoreAllocation + ScoreOffset;
      Overflowed = true;
      CurrentCapacity = NewCap;

      //print out a notice/error message
      //Welp this may be a performance disaster if this is happening too much
      LOG_MESSAGE("Overflowed ReadyList capacity. Old Cap:%d, New Cap:%d, Primary Cap:%d, Prev Overflowed:%B", OldCap, NewCap, PrimaryBufferCapacity, PrevOverflowed);
    }

    //add the instruction to the ready list
    InstrBase[CurrentSize] = Entry.InstId;
    ReadyOnBase[CurrentSize] = Entry.ReadyOn;
    HeurBase[CurrentSize] = Entry.Heuristic;
    ScoreBase[CurrentSize] = Entry.Score;
    ++CurrentSize;
  #endif
}

// We copy the instruction at the end of the array to the instruction at the target index
// then we decrement the Ready List's CurrentSize
// This function has undefined behavior if CurrentSize == 0
__host__ __device__
ACOReadyListEntry ACOReadyList::removeInstructionAtIndex(InstCount Indx) {
  assert(CurrentSize <= 0 || Indx >= CurrentSize || Indx < 0);
  #ifdef __CUDA_ARCH__
    ACOReadyListEntry E{InstrBase[Indx*numThreads_ + GLOBALTID], 
                        ReadyOnBase[Indx*numThreads_ + GLOBALTID], 
                        HeurBase[Indx*numThreads_ + GLOBALTID], 
                        ScoreBase[Indx*numThreads_ + GLOBALTID]};
    InstCount EndIndx = --CurrentSize;
    InstrBase[Indx*numThreads_ + GLOBALTID] = InstrBase[EndIndx*numThreads_ + GLOBALTID];
    ReadyOnBase[Indx*numThreads_ + GLOBALTID] = ReadyOnBase[EndIndx*numThreads_ + GLOBALTID];
    HeurBase[Indx*numThreads_ + GLOBALTID] = HeurBase[EndIndx*numThreads_ + GLOBALTID];
    ScoreBase[Indx*numThreads_ + GLOBALTID] = ScoreBase[EndIndx*numThreads_ + GLOBALTID];
    return E;
  #else
    ACOReadyListEntry E{InstrBase[Indx], ReadyOnBase[Indx], HeurBase[Indx], ScoreBase[Indx]};
    InstCount EndIndx = --CurrentSize;
    InstrBase[Indx] = InstrBase[EndIndx];
    ReadyOnBase[Indx] = ReadyOnBase[EndIndx];
    HeurBase[Indx] = HeurBase[EndIndx];
    ScoreBase[Indx] = ScoreBase[EndIndx];
    return E;
  #endif
}

void ACOReadyList::AllocDevArraysForParallelACO(int numThreads) {
  size_t memSize;
  numThreads_ = numThreads;

  // Alloc dev array for dev_IntAllocation
  memSize = sizeof(InstCount*) * CurrentCapacity * numThreads_ * 2;
  gpuErrchk(cudaMallocManaged(&dev_IntAllocation, memSize));

  // Alloc dev array for dev_HeurAllocation
  memSize = sizeof(HeurType*) * CurrentCapacity * numThreads_;
  gpuErrchk(cudaMallocManaged(&dev_HeurAllocation, memSize));

  // Alloc dev array for dev_ScoreAllocation
  memSize = sizeof(pheromone_t*) * CurrentCapacity * numThreads_;
  gpuErrchk(cudaMallocManaged(&dev_ScoreAllocation, memSize));

  //build shortcut pointers
  InstrBase = dev_IntAllocation;
  ReadyOnBase = dev_IntAllocation + CurrentCapacity*numThreads;
  HeurBase = dev_HeurAllocation;
  ScoreBase = dev_ScoreAllocation;

  // prefetch memory used with cudaMallocManaged
  memSize = sizeof(InstCount*) * numThreads_ * 2;
  gpuErrchk(cudaMemPrefetchAsync(dev_IntAllocation, memSize, 0));

  memSize = sizeof(HeurType*) * numThreads_;
  gpuErrchk(cudaMemPrefetchAsync(dev_HeurAllocation, memSize, 0));

  memSize = sizeof(pheromone_t*) * numThreads_;
  gpuErrchk(cudaMemPrefetchAsync(dev_ScoreAllocation, memSize, 0));

  /*// Alloc dev array for dev_IntAllocation
  memSize = sizeof(InstCount*) * numThreads_ * 2;
  gpuErrchk(cudaMallocManaged(&dev_IntAllocation, memSize));

  // Alloc dev array for dev_HeurAllocation
  memSize = sizeof(HeurType*) * numThreads_;
  gpuErrchk(cudaMallocManaged(&dev_HeurAllocation, memSize));

  // Alloc dev array for dev_ScoreAllocation
  memSize = sizeof(pheromone_t*) * numThreads_;
  gpuErrchk(cudaMallocManaged(&dev_ScoreAllocation, memSize)); */
}

void ACOReadyList::CopyPointersToDevice(ACOReadyList *dev_acoRdyLst, int numThreads) {
  size_t memSize;

  // copy over arrays
  memSize = sizeof(InstCount*) * numThreads * 2;
  for (int i = 0; i < numThreads; i++) {
    gpuErrchk(cudaMemcpy(&dev_acoRdyLst->dev_IntAllocation[i], IntAllocation, memSize,
	  	         cudaMemcpyHostToDevice));
  }
  memSize = sizeof(HeurType*) * numThreads;
  for (int i = 0; i < numThreads; i++) {
    gpuErrchk(cudaMemcpy(&dev_acoRdyLst->dev_HeurAllocation[i], HeurAllocation, memSize,
	  	         cudaMemcpyHostToDevice));
  }
  memSize = sizeof(pheromone_t*) * numThreads;
  for (int i = 0; i < numThreads; i++) {
    gpuErrchk(cudaMemcpy(&dev_acoRdyLst->dev_ScoreAllocation[i], ScoreAllocation, memSize,
	  	         cudaMemcpyHostToDevice));
  }

  // Alloc elmnts for each array
  InstCount *temp_intArr;
  memSize = sizeof(InstCount) * CurrentCapacity * numThreads * 2;
  gpuErrchk(cudaMalloc(&temp_intArr, memSize));

  HeurType *temp_HeurArr;
  memSize = sizeof(HeurType) * CurrentCapacity * numThreads;
  gpuErrchk(cudaMalloc(&temp_HeurArr, memSize));

  pheromone_t *temp_scoreArr;
  memSize = sizeof(pheromone_t) * CurrentCapacity * numThreads;
  gpuErrchk(cudaMalloc(&temp_scoreArr, memSize));

  // assign a chunk of each large array to each array
  for (int i = 0; i < numThreads; i++) {
    dev_acoRdyLst->dev_IntAllocation[i] = temp_intArr[i*CurrentCapacity];
    dev_acoRdyLst->dev_IntAllocation[i + numThreads] = temp_intArr[i*CurrentCapacity + CurrentCapacity*numThreads];
    dev_acoRdyLst->dev_HeurAllocation[i] = temp_HeurArr[i*CurrentCapacity];
    dev_acoRdyLst->dev_ScoreAllocation[i] = temp_scoreArr[i*CurrentCapacity];
  }

  //build shortcut pointers
  InstrBase = dev_IntAllocation;
  ReadyOnBase = dev_IntAllocation + CurrentCapacity*numThreads;
  HeurBase = dev_HeurAllocation;
  ScoreBase = dev_ScoreAllocation;

  // prefetch memory used with cudaMallocManaged
  memSize = sizeof(InstCount*) * numThreads_ * 2;
  gpuErrchk(cudaMemPrefetchAsync(dev_IntAllocation, memSize, 0));

  memSize = sizeof(HeurType*) * numThreads_;
  gpuErrchk(cudaMemPrefetchAsync(dev_HeurAllocation, memSize, 0));

  memSize = sizeof(pheromone_t*) * numThreads_;
  gpuErrchk(cudaMemPrefetchAsync(dev_ScoreAllocation, memSize, 0));
}

void ACOReadyList::FreeDevicePointers() {
  cudaFree(dev_IntAllocation);
  cudaFree(dev_HeurAllocation);
  cudaFree(dev_ScoreAllocation);
}