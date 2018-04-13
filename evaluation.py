from numpy import array
from numpy import argmax
from nltk.translate.bleu_score import corpus_bleu
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length, idx2word):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = idx2word[yhat]
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length, idx2word):
    scores = {}
    
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length, idx2word)
        # store actual and predicted
        actual.append([desc.split()])
        predicted.append(yhat.split())
    # calculate BLEU score
    bleu = corpus_bleu(actual, predicted)
    scores['bleu'] = bleu
    
    perwords=[]
    for i in range(len(predicted)):
        thisScore=0
        for word in actual[i][0]:
            if word in predicted[i]:
                thisScore+=1
        thisScore/=len(actual[i][0])
        perwords.append(thisScore)
    
    scores['perword'] = sum(perwords)/len(perwords)
    
    return scores

def epoch_eval(name, model,n_epochs, descriptions, photos, tokenizer, max_length,idx2word,step=1):
    test_perword=[]
    test_bleu=[]
    ax = []
    for epoch in range(0,n_epochs,step):
        ax.append(epoch)
        print('Epoch : {}'.format(epoch))
        model.load_weights(name+'Epoch'+str(epoch)+'.h5')
        res = evaluate_model(model,descriptions, photos, tokenizer, max_length, idx2word)
        test_perword.append(res['perword'])
        test_bleu.append(res['bleu'])
    #show_test(test_perword,test_bleu,ax)
    return test_perword,test_bleu,ax

def show_test(test_perword,test_bleu,ax):
    plt.plot(ax,test_perword)
    plt.plot(ax,test_bleu)
    plt.title('model accuracy')
    plt.ylabel('test accuracy')
    plt.xlabel('epoch')
    plt.legend(['perword','bleu'], loc='upper left')
    plt.show()
        
        