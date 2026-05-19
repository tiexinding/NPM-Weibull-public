# DATABASE_v9_1 sanity verify report

### pythia-70m (Pythia 70m, MHA-merged)
  records: 24; tokens: 300B; QK-Norm: False
  k_median(qkv) = 1.0488, R2(qkv) = 0.9981, low-R2: 1/6
  k_median(gate)=n/a, k_median(up)=n/a, k_median(down)=n/a

### pythia-160m (Pythia 160m, MHA-merged)
  records: 48; tokens: 300B; QK-Norm: False
  k_median(qkv) = 1.0982, R2(qkv) = 0.9995, low-R2: 0/12
  k_median(gate)=n/a, k_median(up)=n/a, k_median(down)=n/a

### pythia-410m (Pythia 410m, MHA-merged)
  records: 96; tokens: 300B; QK-Norm: False
  k_median(qkv) = 1.1466, R2(qkv) = 0.9989, low-R2: 0/24
  k_median(gate)=n/a, k_median(up)=n/a, k_median(down)=n/a

### pythia-1b (Pythia 1B, MHA-merged)
  records: 64; tokens: 300B; QK-Norm: False
  k_median(qkv) = 1.1758, R2(qkv) = 0.9986, low-R2: 0/16
  k_median(gate)=n/a, k_median(up)=n/a, k_median(down)=n/a

### pythia-6.9b (Pythia 6.9B, MHA-merged)
  records: 128; tokens: 300B; QK-Norm: False
  k_median(qkv) = 1.1726, R2(qkv) = 0.9990, low-R2: 0/32
  k_median(gate)=n/a, k_median(up)=n/a, k_median(down)=n/a

### olmo-1-7b (OLMo-1 7B, MHA-separate)
  records: 224; tokens: 2.5T; QK-Norm: False
  k_median(q)  = 0.8123, R2(q)  = 0.9987, low-R2: 12/32
  k_median(k)  = 0.7601, R2(k)  = 0.9986, low-R2: 13/32
  k_median(v)  = 1.0601, R2(v)  = 0.9971
  k_median(o)  = 1.0409, R2(o)  = 0.9972
  k_median(gate)=1.2010, k_median(up)=1.2039, k_median(down)=1.2041

### olmo-2-7b (OLMo-2 7B, MHA-separate)
  records: 224; tokens: 5T; QK-Norm: True
  k_median(q)  = 0.9895, R2(q)  = 0.9909, low-R2: 2/32
  k_median(k)  = 0.9716, R2(k)  = 0.9936, low-R2: 7/32
  k_median(v)  = 1.1930, R2(v)  = 0.9981
  k_median(o)  = 1.1958, R2(o)  = 0.9980
  k_median(gate)=1.1976, k_median(up)=1.2032, k_median(down)=1.2031

### llama-3-8b (Llama-3 8B, GQA-4:1)
  records: 224; tokens: 15T; QK-Norm: False
  k_median(q)  = 1.1352, R2(q)  = 0.9995, low-R2: 0/32
  k_median(k)  = 1.1462, R2(k)  = 0.9994, low-R2: 0/32
  k_median(v)  = 1.1710, R2(v)  = 0.9985
  k_median(o)  = 1.1841, R2(o)  = 0.9985
  k_median(gate)=1.1890, k_median(up)=1.1971, k_median(down)=1.1931

### mistral-7b (Mistral 7B, GQA-4:1)
  records: 224; tokens: 8T; QK-Norm: False
  k_median(q)  = 1.1485, R2(q)  = 0.9993, low-R2: 1/32
  k_median(k)  = 1.1291, R2(k)  = 0.9996, low-R2: 0/32
  k_median(v)  = 1.1702, R2(v)  = 0.9984
  k_median(o)  = 1.1902, R2(o)  = 0.9983
  k_median(gate)=1.1947, k_median(up)=1.1964, k_median(down)=1.1926

### qwen2.5-7b (Qwen2.5 7B, GQA-7:1)
  records: 196; tokens: 18T; QK-Norm: False
  k_median(q)  = 1.1328, R2(q)  = 0.9993, low-R2: 0/28
  k_median(k)  = 1.1033, R2(k)  = 0.9998, low-R2: 0/28
  k_median(v)  = 1.1430, R2(v)  = 0.9987
  k_median(o)  = 1.1665, R2(o)  = 0.9989
  k_median(gate)=1.1904, k_median(up)=1.1888, k_median(down)=1.1830

### qwen2.5-14b (Qwen2.5 14B, GQA-5:1)
  records: 189; tokens: 18T; QK-Norm: False
  k_median(q)  = 1.1598, R2(q)  = 0.9989, low-R2: 0/27
  k_median(k)  = 1.1346, R2(k)  = 0.9993, low-R2: 0/27
  k_median(v)  = 1.1636, R2(v)  = 0.9985
  k_median(o)  = 1.1841, R2(o)  = 0.9985
  k_median(gate)=1.1909, k_median(up)=1.1914, k_median(down)=1.1885

### qwen3-8b (Qwen3 8B, GQA-4:1)
  records: 252; tokens: 36T; QK-Norm: True
  k_median(q)  = 1.1623, R2(q)  = 0.9991, low-R2: 0/36
  k_median(k)  = 1.1539, R2(k)  = 0.9992, low-R2: 0/36
  k_median(v)  = 1.1581, R2(v)  = 0.9986
  k_median(o)  = 1.1802, R2(o)  = 0.9986
  k_median(gate)=1.1872, k_median(up)=1.1886, k_median(down)=1.1846
