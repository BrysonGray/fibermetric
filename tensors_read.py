import nibabel as nib

tensors_fname = '/home/brysongray/data/BrainSuiteTutorialDWI/2523412.dwi.RAS.correct.T1_coord.eig.nii'

with open(tensors_fname, 'rb') as f:
    print(f.read(1)[0])
    word1 = f.read(1)
    word2 = f.read(2)
    