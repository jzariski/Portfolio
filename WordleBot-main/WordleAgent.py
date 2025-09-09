import numpy as np
import random
from sklearn import linear_model
import random
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_dict():
    dictionary = open('wordle_dictionary.txt', 'r')
    dictSet = set()
    for item in dictionary:
        dictSet.add(str(item)[0:len(item) - 1]) 
    dictSet.discard('zona')
    dictSet.add('zonal')   
    return dictSet

def get_letter_count(wordSet):
    letter_count = dict()
    for word in wordSet:
        for letter in word:
            if letter not in letter_count.keys():
                letter_count[letter] = 1
            else:
                letter_count[letter] = letter_count[letter] + 1
    return letter_count

def train_regress(prob, NN=False):
    ## Build training targets for the regressor. We map each guess string to
    ## the *average remaining-candidate count* observed across many simulated states.
    ## Predicting "expected remaining" is better aligned with fast Wordle solving
    ## than maximizing "discards" and is numerically stable for regression.

    dictionary = {}

    ## Building the training set
    for i in range(10000):
        agent = WordleAgent()
        pairs = agent.problem_interface_get_info(prob)
        for item in pairs:
            string = item[0]
            # Prefer explicit 'remaining' if we added it; else derive from discards
            remaining = item[2] if len(item) > 2 else None
            if remaining is None:
                discards = item[1]
                remaining = max(1, len(agent.original_dictionary) - discards)

            # Guard against weird values
            if remaining is None:
                continue
            # clamp to positive range
            remaining = max(1, int(remaining))

            dictionary.setdefault(string, []).append(remaining)

    # Vectorize
    X, Y = [], []
    for string, vals in dictionary.items():
        if not vals:
            continue
        avg_remaining = float(sum(vals)) / max(len(vals), 1)
        # Keep strictly positive & finite
        if not (avg_remaining > 0 and avg_remaining < 1e9):
            continue
        X.append(WordleAgent().encodeWord(string))
        Y.append(avg_remaining)

    X = np.asarray(X)
    Y = np.asarray(Y, dtype=float)

    # Final guards (avoid empty set / NaNs)
    mask = np.isfinite(Y) & (Y > 0)
    X = X[mask]
    Y = Y[mask]


    if not NN:
        # --- Use squared error to avoid Poisson base_score issues ---
        from xgboost import XGBRegressor
        regr = XGBRegressor(
            n_estimators=900,
            learning_rate=0.06,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_alpha=1e-3,
            reg_lambda=1.0,
            objective="reg:squarederror", 
            tree_method="hist",
        )
        regr.fit(X, Y)
        return regr
    else:
        ## Scale the target for continuity
        y_train = np.log1p(Y)  ## network learns log(1 + remaining)

        in_dim = X.shape[1]
        inputs = keras.Input(shape=(in_dim,), name="features")
        x = layers.Dense(64, activation="relu")(inputs)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs, outputs)

        opt = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])

        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ]

        ## Fit (uses a random val split)
        model.fit(
            X, y_train,
            validation_split=0.15,
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=cbs,
            shuffle=True,
        )
        return model

class WordleAgent():

    def __init__(self):
        self.original_dictionary = get_dict()
        self.current_dictionary = self.original_dictionary.copy()
        self.green_tuples = set()
        self.yellow_tuples = set()
        self.grey_letters = set()

    
    def augment_information(self, tuples):
        greens, yellows, greys = tuples
        for green in greens:
            self.green_tuples.add(green)
        for yellow in yellows:
            self.yellow_tuples.add(yellow)
        for grey in greys:
            self.grey_letters.add(grey)
        
    def augment_possible_answers(self, guess):
        total = len(self.current_dictionary)
        newDict = self.current_dictionary.copy()
        for item in self.current_dictionary:
            remove = False
            val = 0
            for tup in self.green_tuples:
                letter, index = tup[0], tup[1]
                if item[index] != letter:
                    remove = True
                    val += 2
            
            for tup in self.yellow_tuples:
                letter, index = tup[0], tup[1]
                if letter not in item:
                    remove = True
                    val += 5

            for letter in self.grey_letters:
                if letter in item:
                    newDict.discard(item)
                    remove = True
                    val += 10

            if remove:
                newDict.discard(item)
        newDict.discard(guess)
        self.current_dictionary= newDict
        return val * (1-(total/len(self.original_dictionary)))


    ## These guess functions only choose from a space of possible answers
    ## Could limit the amount of new information that can be obtained

    def make_guess_random(self):
        guess_list = list(self.current_dictionary.copy())
        return random.choice(guess_list)
    
    def make_guess_choice(self):
        totalLetters = 5 * len(self.original_dictionary)
        max, maxWord = -1 * np.inf, None
        letter_count = get_letter_count(self.current_dictionary)
        for word in self.current_dictionary:
            word_score = 0
            for letter in word:
                word_score += letter_count[letter] / totalLetters
            if word_score > max:
                max = word_score
                maxWord = word
        return maxWord


    def problem_interface_not_human(self, wordleProblem):
        print('Welcome to WordleBot')
        #guess = self.make_guess_random()
        guess = self.make_guess_choice()
        guessCount = 1
        while not wordleProblem.isCorrect(guess):
            print('Guess: ', guess)
            tuples = wordleProblem.get_information(guess)
            self.augment_information(tuples)
            self.augment_possible_answers(guess)
            guess = self.make_guess_random()
            guessCount += 1
        print('Guess: ', guess)
        print(guess, ' is correct!')
        print('Solved in ', guessCount, ' guesses!')
        return guessCount
    
    def problem_interface_no_print(self, wordleProblem):
        guess = self.make_guess_random()
        guessCount = 1
        while not wordleProblem.isCorrect(guess):
            tuples = wordleProblem.get_information(guess)
            self.augment_information(tuples)
            self.augment_possible_answers(guess)
            guess = self.make_guess_random()
            guessCount += 1
        return guessCount
    
    def encodeWord(self, guess):
        arrays = np.zeros(1)
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        for letter in guess:
            letter_num = alphabet.index(letter)
            one_hot = np.zeros(26) ## One-hot encoding our wordle words
            one_hot[letter_num] = 1
            arrays = np.concatenate((arrays, one_hot), axis =0)
        return arrays


    
    def problem_interface_get_info(self, wordleProblem):
        #print('Welcome to WordleBot')
        guess = self.make_guess_random()
        guessCount = 1
        tups = set()
        while not wordleProblem.isCorrect(guess):
            #print('Guess: ', guess)
            tuples = wordleProblem.get_information(guess)
            self.augment_information(tuples)
            discards = self.augment_possible_answers(guess)
            guess = self.make_guess_random()
            guessCount += 1
            tups.add((guess, discards, len(self.current_dictionary))) ## Important info
        #print('Guess: ', guess)
        #print(guess, ' is correct!')
        #print('Solved in ', guessCount, ' guesses!')
        return tups
    
    def make_guess_regress(self, regr):
        best_guess, best_val = None, float('inf')
        for item in self.current_dictionary:
            word = self.encodeWord(item).reshape(1,-1)
            if isinstance(getattr(regr, "keras_model", regr), tf.keras.Model):
                val = float(regr.predict(word, verbose=0))
            else:
                val = float(regr.predict(word))
            if val < best_val:
                best_guess, best_val = item, val

        return best_guess

    def problem_interface_regress(self, wordleProblem, regr):        
        guess = self.make_guess_regress(regr)
       
        guessCount = 1
        while not wordleProblem.isCorrect(guess):
            tuples = wordleProblem.get_information(guess)
            self.augment_information(tuples)
            self.augment_possible_answers(guess)
            num = random.random()
            guess = self.make_guess_regress(regr)
            guessCount += 1
        return guessCount
