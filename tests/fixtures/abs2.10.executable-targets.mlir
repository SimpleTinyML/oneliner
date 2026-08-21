#executable_target_static = #hal.executable.target<"llvm-cpu", "static", {cpu = "", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, link_embedded = false, link_static = true, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, static_library_output = "abs2.o", target_triple = "x86_64-unknown-linux-gnu"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#device_target_local = #hal.device.target<"local", [#executable_target_static]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_local
  hal.executable private @main_dispatch_0 {
    hal.executable.variant public @static target(#executable_target_static) {
      hal.executable.export public @main_dispatch_0_elementwise_2_f32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        %c1_1 = arith.constant 1 : index
        hal.return %c1, %c1_0, %c1_1 : index, index, index
      } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
      builtin.module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
        llvm.func @main_dispatch_0_elementwise_2_f32(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {
          %0 = llvm.mlir.constant(0 : i32) : i32
          %1 = llvm.mlir.constant(64 : index) : i64
          %2 = llvm.mlir.constant(true) : i1
          %3 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>
          %4 = llvm.extractvalue %3[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>
          %5 = llvm.load %4 : !llvm.ptr -> !llvm.ptr
          llvm.intr.assume %2 ["align"(%5, %1 : !llvm.ptr, i64)] : i1
          %6 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>
          %7 = llvm.extractvalue %6[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>
          %8 = llvm.getelementptr %7[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
          %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr
          llvm.intr.assume %2 ["align"(%9, %1 : !llvm.ptr, i64)] : i1
          %10 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> vector<2xf32>
          %11 = llvm.intr.fabs(%10) : (vector<2xf32>) -> vector<2xf32>
          llvm.store %11, %9 {alignment = 4 : i64} : vector<2xf32>, !llvm.ptr
          llvm.return %0 : i32
        }
      }
    }
  }
  util.func public @main(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @main(%input0: tensor<2xf32>) -> (%output0: tensor<2xf32>)"}} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c2]) type(%element_type_f32) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<2xf32> in !stream.resource<external>{%c8}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%c8} => !stream.timepoint
    %1 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%0 as %arg1: !stream.resource<external>{%c8}, %result as %arg2: !stream.resource<external>{%c8}) {
      stream.cmd.dispatch @main_dispatch_0::@static::@main_dispatch_0_elementwise_2_f32 {
        ro %arg1[%c0 for %c8] : !stream.resource<external>{%c8},
        wo %arg2[%c0 for %c8] : !stream.resource<external>{%c8}
      }
    } => !stream.timepoint
    %2 = stream.timepoint.await %1 => %result : !stream.resource<external>{%c8}
    %3 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %2 : tensor<2xf32> in !stream.resource<external>{%c8} -> !hal.buffer_view
    util.return %3 : !hal.buffer_view
  }
}
