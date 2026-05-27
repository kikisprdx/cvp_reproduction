# cvp_reproduction

SVP:
# The self-supervised visual prompts attempt to reverse the adversarial attacks by modifying the input pixels with ℓp-norm perturbations, where the perturbations are optimized via contrastive loss [ 40 ]. For the patch setting, we setup the shape of VP as 32*32*3 for CIFAR-C and 224*224*3 for all ImageNet OOD datasets. For the padding setting, we set the padding size as 1 for CIFAR-10-C and 15 for ImageNet OOD dataset. Take CIFAR data as example, we first initialize a mask with all zeros value with the shape 30*30*3 and set the pad value as 1 with padding size 1 so that the mask after padding is as the same shape of CIFAR data (32*32*3). Then, we multiply the mask with the VP to preserve only the VP located at the position we just pad with 1 value. We can further optimize the VP with mask by adding it with the corrupted samples
dversarial attacks are reversible with natural supervision

unordered list 
