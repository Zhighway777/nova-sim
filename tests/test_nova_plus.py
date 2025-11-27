# import pytest
# from nova_suite.service import nova_plus_service as nova
# from end_to_end.register import TopoName

# column_order_file = "nova_suite/order.txt"
# with open(column_order_file, "r") as f:
#     column_order = f.read().splitlines()


# def test_do_job_train(outdir):
#     train_params = nova.NovaPlusTrainParams(
#         mode="TRAIN",
#         arch=[TopoName.H100],
#         model_names=["gpt175b"],
#         micro_batch_nums=[8],
#         micro_batch_size=[1],
#         precision=[nova.NovaPlusPrecision.FP16],
#         seqlen=[4096]
#     )
#     nova.do_job(train_params, outdir, column_order)


# def test_do_job_inference(outdir):
#     inference_params = nova.NovaPlusInferenceParams(
#         mode="INFERENCE",
#         arch=[TopoName.H100],
#         model_names=["gpt175b"],
#         batchsize=[1],
#         precision=[nova.NovaPlusPrecision.FP16],
#         seqlen_group=[(4096, 2048, 128)]
#     )

#     nova.do_job(inference_params, outdir, column_order)
