// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_ptoas.h"
#include "target/source/codegen_source_base.h"

namespace tvm {
namespace codegen {

runtime::Module BuildTileLangPTOAS(IRModule mod, Target target, std::string platform) {
  using tvm::runtime::Registry;
  CodeGenTileLangPTOAS cg(platform);
  cg.Init();

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangAscendPto: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  std::string host = cg.GetHostFn();
  code = code + "\n=======\n" + host;

  auto res = CSourceModuleCreate(code, "c", {});
  return res;
}

TVM_REGISTER_GLOBAL("target.build.tilelang_ptoas")
    .set_body_typed(BuildTileLangPTOAS);

} // namespace codegen
} // namespace tvm

