from os import listdir

check=set(listdir('Flicker8k_Dataset'))

# extract descriptions for images
def load(filename):
    #return as {'id':'description'}
    with open(filename) as f:
        lines = f.read().splitlines()
    desc = dict()
    for line in lines:
        if len(line) < 2:
            continue
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        #print (image_id)
        if (image_id not in desc) and ((image_id+'.jpg') in check):
            desc[image_id] = image_desc
    return desc

#cleanig
def clean(full_desc):
    import string
    for img_id, desc in full_desc.items():
        desc = ''.join([word.lower() for word in desc])
        # remove punctuation from each token
        desc = ''.join(l for l in desc if l not in string.punctuation)
        # remove single char words
        desc = [word for word in desc.split() if len(word)>1]
        full_desc[img_id] = 'startseq '+' '.join(desc)+' endseq'