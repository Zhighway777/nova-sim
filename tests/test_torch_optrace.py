# import optrace_benchmark as optrace


# def test_torch_optrace_file(outdir):
#     op_trace_path = "tests/resources/simple_optrace.txt"
#     op_trace_path = "tests/resources/simple_optrace.txt"
#     opt = optrace.optrace(op_trace_path)
#     mod = opt.getModule()  # string to object sample

#     with open(f"{outdir}/optrace_file_output.txt", "w") as f:
#         instruct_list = mod.getInstructs()

#     with open(f"{outdir}/optrace_file_output.txt", "w") as f:
#         instruct_list = mod.getInstructs()

#         for instr_dump in instruct_list:
#             f.write(f"{instr_dump}\n")

#             f.write(f"Stream ID: {instr_dump.getStreamID()}\n")       # 流 ID
#             f.write(f"Domain: {instr_dump.getDomain()}\n")            # 操作域
#             f.write("-" * 40 + "\n")

# #test to print into optrace_file_output.txt
# # test_torch_optrace_file("./")


# def test_torch_optrace_create(outdir):
#     scalar_1 = optrace.scalar(1, optrace.datatype.I32)
#     scalar_2 = optrace.scalar("offset", optrace.datatype.I32, "1024")
#     tensor_1 = optrace.tensor(2, optrace.datatype.I32, [2, 3, 4], [12, 4, 1], 512)
#     tensor_2 = optrace.tensor(3, optrace.datatype.I32, [2, 3, 4])

#     struct_1 = optrace.structure(5, "tuple", [scalar_2, tensor_1, scalar_1])
#     struct_2 = optrace.structure(6, "tuple", [struct_1, tensor_1, scalar_1])
#     struct_3 = optrace.structure("paramname", 7, "tuple", [struct_1, tensor_1, scalar_1])

#     instruct_2 = optrace.instruct(
#         "A123", 1, 3, "aten::sample_add", "torch", "2_11", [tensor_1, scalar_1, scalar_2, struct_3], [struct_2]
#     )
