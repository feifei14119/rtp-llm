import torch
import unittest
import os
import numpy as np

def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)

class TestGemmDequantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/libth_transformer.so")
        #torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")
        self.pack_int4s = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
        #self.fused_gemm_dq = torch.ops.gemm_dq_unit_ops.fused_gemm_dq
        self.preprocess_weights_for_mixed_gemm = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
        torch.manual_seed(734876213)

    def woq_groupwise_extract_int4(self, w_packed, uint4_input=False):
      w_packed_int8 = w_packed.T.contiguous().view(torch.uint8)
      w_unpacked_int4 = torch.stack(((w_packed_int8 % 16).view(-1, 1), (w_packed_int8 // 16).view(-1, 1)), dim=1)

      # Unpacked uint4s
      w_unpacked_int4 = w_unpacked_int4.flatten().view(w_packed.shape[1], -1).T.contiguous().int()
      
      print("========= woq_groupwise_extract_int4 ==========")
      print("---------- w_packed ---------------------------")
      print(w_packed)
      print("---------- w_packed.T -------------------------")
      print(w_packed.T)
      print("---------- w_packed.contiguous() --------------")
      print(w_packed.T.contiguous())
      print("---------- w_packed_int8 ----------------------")
      print(w_packed_int8)
      #print("---------- w_unpacked_int4 --------------------")
      #print(w_unpacked_int4)
      print("---------- w_unpacked_int4 --------------------")
      print(w_unpacked_int4)
      print("===============================================")

      if not uint4_input:
          w_unpacked_int4 -= 8
      return w_unpacked_int4    
      
    def woq_assert_colwise_near_eq(self, ref, act):
      bits_in_type = 4
      quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

      # check each column independently
      if ref.shape[0] > 1:
          for col_idx in range(ref.shape[-1]):
              col = ref[:, col_idx]
              max_val = torch.max(col).item()
              atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
              np.testing.assert_allclose(col.cpu().numpy(),
                                         act[:, col_idx].cpu().numpy(),
                                         atol=atol)
      else:
          max_val = torch.max(ref).item()
          atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
          np.testing.assert_allclose(ref.cpu().numpy(),
                                     act.cpu().numpy(),
                                     atol=atol)
    
    def groupwise_gemm_dequant_test_helper(self, compute_type, gemm_ms, gemm_ns, gemm_ks, group_size):
      uint4_input=1
      for gemm_m in gemm_ms:
        for gemm_k in gemm_ks:
          for gemm_n in gemm_ns:
            torch.manual_seed(0)
            activation = torch.rand((gemm_m, gemm_k), dtype=compute_type) * 2 - 1.0 
            qweight_unprocessed = torch.randint(-2**31, 2**31, (gemm_k // 8, gemm_n)).int()
            scale = torch.rand((gemm_k // group_size, gemm_n), dtype=compute_type) * 2
            zeros = torch.rand((gemm_k // group_size, gemm_n), dtype=compute_type) * 2

            qweight_int8 = self.woq_groupwise_extract_int4(qweight_unprocessed, uint4_input).char()
            real_input_weight_int4 = qweight_int8 - uint4_input * 8
            qweight_pack_int4s = self.pack_int4s(real_input_weight_int4)
            qweight_int4x2_interleaved = self.preprocess_weights_for_mixed_gemm(qweight_pack_int4s, torch.quint4x2)

            scale_interleave = scale.repeat_interleave(group_size, dim=0)
            zeros_interleave = zeros.repeat_interleave(group_size, dim=0)

            print("group_size = ", group_size)
            print("mnk = {0:d}, {1:d}, {2:d}".format(gemm_m, gemm_n, gemm_k))
            print("---------- activation ---------------------------")
            print(activation)
            print("---------- qweight_unprocessed ------------------")
            print(qweight_unprocessed)
            print("---------- qweight_int8 -------------------------")
            print(qweight_int8)
            print("---------- qweight_int8 - 8 ---------------------")
            print(real_input_weight_int4)
            print("---------- qweight_pack_int4s -------------------")
            print(qweight_pack_int4s)
            print("---------- qweight_int4x2_interleaved -----------")
            print(qweight_int4x2_interleaved)
            #print("---------- scale --------------------------------")
            #print(scale)
            #print(scale_interleave)
            #print("---------- zeros --------------------------------")
            #print(zeros)
            #print(zeros_interleave)
            
            qweight_uint4 = real_input_weight_int4 + 8
            qweight_int8_half = qweight_uint4.half()
            ref_th_weight = qweight_int8_half * scale_interleave - 8 * scale_interleave
            ref_th_weight += zeros_interleave
            print("=================== TORCH =======================")
            print("---------- qweight_int8 -------------------------")
            print(qweight_int8)
            print("---------- real_input_weight_int4 ---------------")
            print(real_input_weight_int4)
            print("---------- qweight_uint4 ------------------------")
            print(qweight_uint4)
            print("---------- qweight_int8_half --------------------")
            print(qweight_int8_half)
            print("---------- qweight_int8 * scale -----------------")
            print(qweight_int8_half * scale_interleave)
            print("---------- qweight_int8 * scale -----------------")
            print(qweight_int8_half * scale_interleave - 8 * scale_interleave)
            print("---------- ref_th_weight ------------------------")
            print(ref_th_weight)

            #ft_result = self.fused_gemm_dq(activation.cuda(), qweight_int4x2_interleaved.cuda(), scale.cuda(), zero.cuda(), group_size, True)
            
            reference_result = activation.cuda().matmul(ref_th_weight.cuda().to(compute_type))
            self.woq_assert_colwise_near_eq(reference_result, reference_result)
      
    def test_fp16_int4_gemm(self):
      self.groupwise_gemm_dequant_test_helper(torch.float16, 
                                    gemm_ms = [1],
                                    gemm_ns = [8],
                                    gemm_ks = [16],
                                    group_size=4)

if __name__ == '__main__':
    unittest.main()
