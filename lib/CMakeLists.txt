if(${PACKAGE_VERSION} VERSION_LESS "8.0")
  add_llvm_loadable_module(OptSched OBJECT ${OPT_SCHED_SRCS})
else()
  add_llvm_library(OptSched MODULE OBJECT ${OPT_SCHED_SRCS})
endif()

add_llvm_target(
  OptSched Scheduler/aco.cpp
  Scheduler/bb_spill.cpp
  Scheduler/buffers.cpp
  Scheduler/config.cpp
  Scheduler/data_dep.cpp
  Scheduler/enumerator.cpp
  Scheduler/gen_sched.cpp
  Scheduler/graph.cpp
  Scheduler/graph_trans.cpp
  Scheduler/hist_table.cpp
  Scheduler/list_sched.cpp
  Scheduler/logger.cpp
  Scheduler/reg_alloc.cpp
  Scheduler/utilities.cpp
  Scheduler/machine_model.cpp
  Scheduler/random.cpp
  Scheduler/ready_list.cpp
  Scheduler/register.cpp
  Scheduler/relaxed_sched.cpp
  Scheduler/sched_basic_data.cpp
  Scheduler/sched_region.cpp
  Scheduler/stats.cpp
  Wrapper/OptimizingScheduler.cpp
  Wrapper/OptSchedMachineWrapper.cpp
  Wrapper/OptSchedDDGWrapperBasic.cpp
  Wrapper/OptSchedGenericTarget.cpp
  Wrapper/AMDGPU/GCNOptSched.cpp
  Wrapper/AMDGPU/OptSchedGCNTarget.cpp
  Wrapper/AMDGPU/OptSchedDDGWrapperGCN.cpp

  STATIC
)

target_include_directories(LLVMOptSched PRIVATE ${OPTSCHED_TARGET_DEPS})

add_dependencies(OptSched ${OPT_SCHED_TARGET_DEPS})


#target_compile_definitions(LLVMOptSched PRIVATE INSERT_ON_STEPFRWRD)
target_include_directories(LLVMOptSched PRIVATE ${OPTSCHED_TARGET_DEPS})
target_link_libraries(OptSched -L/home/paul/ROCm/opencl/build/lib/libamdocl64.so)
