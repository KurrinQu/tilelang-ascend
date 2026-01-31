// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_ptoas.h"

namespace tvm {
namespace codegen {

runtime::Module BuildTileLangPTOAS(IRModule mod, Target target, std::string platform) {
  using tvm::runtime::Registry;
  CodeGenTileLangPTOAS cg(platform);
  cg.Init();

  Array<String> function_names;

  std::string code = cg.Finish();

  return CSourceModuleCreate(code, "c", function_names);
}

TVM_REGISTER_GLOBAL("target.build.tilelang_ptoas")
    .set_body_typed(BuildTileLangPTOAS);

} // namespace codegen
} // namespace tvm

