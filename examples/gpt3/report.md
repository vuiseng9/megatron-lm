



Setup Requirements: minimum 8xA100-80GB, the results following were obtained with 8xH100-80GB sxm5

no checkpoint savings



SXM5 ID!
$ nvidia-smi -q | grep -E "Product Name|Product Architecture|GPU Part Number|Form Factor"
    Product Name                          : NVIDIA H100 80GB HBM3
    Product Architecture                  : Hopper
    GPU Part Number                       : 2330-885-A1
    Product Name                          : NVIDIA H100 80GB HBM3

(For reference, the PCIe version would be 2330-890-xx) claude said