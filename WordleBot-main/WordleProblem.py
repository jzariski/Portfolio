import numpy as np
import random
import WordleAgent as wa
from WordleAgent import WordleAgent
from sklearn import linear_model


def get_dict():
    dictionary = open('wordle_dictionary.txt', 'r')
    dictSet = set()
    for item in dictionary:
        dictSet.add(str(item)[0:len(item) - 1])  
    dictSet.discard('zona')
    dictSet.add('zonal')  
    return dictSet

class WordleProblem():

    def __init__(self):
        self.original_dictionary = get_dict()
        self.dict_as_list = list(self.original_dictionary.copy())
        self.answer = random.choice(self.dict_as_list)

    def isCorrect(self, guess):
        return guess == self.answer
    
    def get_information(self, guess):
        if self.isCorrect(guess):
            print(guess + ' is the correct answer!')
        else:
            green_tuples, yellow_tuples, grey_letters = set(), set(), set()
            for i in range(5):
                if guess[i] == self.answer[i]:
                    green_tuples.add((guess[i], i))
                elif guess[i] in self.answer:
                    yellow_tuples.add((guess[i], i))
                else:
                    grey_letters.add(guess[i])
        return green_tuples, yellow_tuples, grey_letters


def scheme(model):
    guesses1 = 0
    guesses2 = 0

    n = 10
    
    for i in range(n):
        print('Test Number', i)
        prob = WordleProblem()
        agent1 = WordleAgent()
        agent2 = WordleAgent()

        guesses1 += agent1.problem_interface_no_print(prob)

        guesses2 += agent2.problem_interface_regress(prob, model)



    return (guesses1 / n), (guesses2 / n)



model = wa.train_regress(WordleProblem(), True)
print('Model Trained')

val1, val2 = scheme(model)
print('Random gets score of: ', val1)

print('Regression gets score of: ', val2)
