"""
Author: Joshua Jansen Van Vueren
Date: 12 Aug 2020
Desc: Basic Class to create Ngram LM's
"""

import math

class NgramPy:
    """
    Simple n-gram that can do rudimentary n-gram tasks
    
    file_name : name of text file from which to generate n-grams

    level: maximum level of n-grams to count

    use_oov : whether to use a OOV token or ...? 
    
    grams : dict
    Contains all the relevant n-grams from the provided corpus...
    """
    def __init__(self,file_name,level=2,use_oov=False):
        self.n_gram_file = file_name
        self.level = level
        self.use_oov = use_oov
        self.bigrams = {}
        self.bigram_counts = {}
        self.unigram_unique_branches = {}
        self.unigram_lambda_coeff = {}
        self.unigrams = {}
        self.word_counts = {}
        self.vocab = ([])

        self.grams = {}

        for i in range(1,level + 1):
            self.grams[i] = {}

    def count(self):
        with open(self.n_gram_file) as f:
            dataset_len = 0
            n = 2
            for line in f:
                words = line.split()
                dataset_len += len(words) - 1

                for word in words:
                    if word not in self.vocab:
                        self.vocab.append(word)

                for level in range(1,n + 1):
                    line_grams = [tuple(words[j:j+level]) for j in range(len(words)-level+1)]

                    for gram in line_grams:
                        if gram in self.grams[level]:
                            self.grams[level][gram] += 1
                        else:
                            self.grams[level][gram] = 1

                
                #prev_word = ""
                #word = ""
                #word_complete = False
                #for ch in line:
                #    # add unique words to a vocab
                #    if ch == " " or ch == "\n":
                #        if word not in self.vocab:
                #            self.vocab.append(word)
                #
                #        word_complete = True
                #        dataset_len += 1
                #    else:
                #        word_complete = False
                #        word += ch
                #
                #    # add words to grams dict
                #    if (word_complete):
                #        if word in self.word_counts:
                #            self.word_counts[word] += 1
                #        else:
                #            self.word_counts[word] = 1
                #
                #    # add bigrams to grams dict
                #    if (word_complete and prev_word != ""):
                #        if str(prev_word + " " + word) in self.bigram_counts:
                #            self.bigram_counts[str(prev_word + " " + word)] += 1
                #        else:
                #            self.bigram_counts[str(prev_word + " " + word)] = 1
                #
                #            # get unique branches
                #            # - bigram is novel
                #            # - if history is novel set count to 1
                #            # - else add one new branch to the history
                #
                #            if prev_word in self.unigram_unique_branches:
                #                self.unigram_unique_branches[prev_word] += 1
                #            else:
                #                self.unigram_unique_branches[prev_word] = 1
                #
                #    # reset word
                #    if word_complete:
                #        prev_word = word
                #        word = ""

        level = len(self.grams)
        for _ in range(len(self.grams)):
            if level == 1:
                # for unigrams the probability is normalised by the dataset
                for unigram in self.grams[level]:
                    self.grams[level][unigram] = math.log10(self.grams[level][unigram] * 1.0 / dataset_len)
            else:
                # otherwise normalise by the history
                for n_gram in self.grams[level]:
                    self.grams[level][n_gram] = math.log10(self.grams[level][n_gram] * 1.0 / self.grams[level-1][n_gram[:-1]])
            level -= 1

        self.grams[1][("<s>",)] = -99

    def witten_bell_smooth(self):
        wb = {}

        #
        # TODO: the lambda coeff is wrong ...
        #

        for key in self.bigrams:
            lambda_coeff = 0
            hist = key.split()[0]

            # occurances of the history
            #num_occur = self.word_counts[hist]
            num_occur = self.bigram_counts[key]

            # unique words followed on from history
            uniq_followers = self.unigram_unique_branches[hist]

            lambda_coeff = 1.0 - ((uniq_followers * 1.0) / (uniq_followers + num_occur))
            self.unigram_lambda_coeff[hist] = math.log10(lambda_coeff)

            wb[key] = math.log10(lambda_coeff*(10**(self.bigrams[key])) + (1-lambda_coeff)*(10**self.unigrams[key.split()[1]]))

        self.bigrams = wb


    def get_n_grams(self):
        return self.bigrams

    def calc_perp(self,file_name):
        log_prob = 0.0
        test_set_len = 0

        # calculate perplexity of provided file
        with open(file_name) as test_set:
            for line in test_set:
                words = line.split()
                test_set_len += len(words) - 1 # no start token
                line_probs = [0. for i in range(len(words)+1)]
                for level in range(self.level,0,-1):
                    line_grams = [tuple(words[j:j+level]) for j in range(len(words)-level+1)]

                    for i,line_gram in enumerate(line_grams):
                        if line_probs[i] == 0 and line_gram in self.grams[level]:
                            # but if it tries to back off to a unigram <s> or </s> stop it
                            if not (level == 1 and line_gram == ('<s>',) or line_gram == ('</s>')):
                                line_probs[i] = self.grams[level][line_gram]

                log_prob += sum(line_probs)

        print(log_prob) 
        return 10**((-1.0/test_set_len)*log_prob)

    def calc_seq_perp(self,seq,debug_level=1):
        if debug_level == 1:
            log_prob = 0.0
            test_set_len = 0

            # calculate perplexity of provided file
            words = seq.split()
            test_set_len += len(words) - 1 # no start token
            line_probs = [0. for i in range(len(words)+1)]
            for level in range(self.level,0,-1):
                line_grams = [tuple(words[j:j+level]) for j in range(len(words)-level+1)]

                for i,line_gram in enumerate(line_grams):
                    if line_probs[i] == 0 and line_gram in self.grams[level]:
                        # but if it tries to back off to a unigram <s> or </s> stop it
                        if not (level == 1 and line_gram == ('<s>',) or line_gram == ('</s>')):
                            line_probs[i] = self.grams[level][line_gram]

            log_prob += sum(line_probs)

            return 10**((-1.0/len(seq))*log_prob)

        elif debug_level == 2:
            log_prob = 0.0
            seq_len = -1

            # calculate perplexity of provided file
            words = seq.split()
            seq_len += len(words) - 1 # no start token

            line_probs = [0. for i in range(len(words)-1)]
            used_grams = ["" for i in range(len(words)-1)]
            used_level = [0. for i in range(len(words)-1)]

            for level in range(self.level,0,-1):
                line_grams = [tuple(words[j:j+level]) for j in range(len(words)-level+1)]

                if level != 1:
                    for i,line_gram in enumerate(line_grams):
                        if line_probs[i] == 0 and line_gram in self.grams[level]:
                            # but if it tries to back off to a unigram <s> or </s> stop it
                            if not (level == 1 and line_gram == ('<s>',) or line_gram == ('</s>')):
                                line_probs[i] = self.grams[level][line_gram]
                                used_grams[i] = line_gram
                                used_level[i] = level
                else:
                    for i,line_gram in enumerate(line_grams[:-1]):
                        if line_probs[i] == 0 and line_gram in self.grams[level]:
                            # but if it tries to back off to a unigram <s> or </s> stop it
                            if not (level == 1 and line_gram == ('<s>',) or line_gram == ('</s>')):
                                line_probs[i] = self.grams[level][line_gram]
                                used_grams[i] = line_gram
                                used_level[i] = level

            log_prob += sum(line_probs)
            ppl = 10**((-1.0/seq_len)*log_prob)
            with open("debug_seqs","w+") as f:
                f.write(seq + "\n")

                for i,gram in enumerate(used_grams):
                    f.write("\t p(" + str(gram) + ") \t {" + str(used_level[i]) + " gram } \t = "
                             + str(line_probs[i]) + "\t [" + str(10**line_probs[i]) + "]\n")

                f.write(str(seq_len) + " words; logprob = " + str(log_prob) + " ppl = " +  str(ppl))
            

            return 10**((-1.0/seq_len)*log_prob)

    def export_n_gram_model(self,export_file_name="lm.txt"):
        with open("lm.txt","w+") as f:
            f.write("\\data\\" + "\n")
            
            level = 1
            for _ in range(len(self.grams)):
                f.write ("ngram " + str(level) + "=" + str(len(self.grams[level])) + "\n")
                level += 1

            f.write("\n")
            level = 1
            for _ in range(len(self.grams)):
                f.write("\\" + str(level) + "-grams:\n")
                for n_gram in sorted(self.grams[level]):
                    f.write(str(self.grams[level][n_gram]) + "\t" + str(n_gram) + "\n")
                level += 1
                f.write("\n")

    #################################################################
    ##################### UTILITY FUNCTIONS #########################
    #################################################################


if __name__ == '__main__':
    # load dataset
    test_n_gram = NgramPy("dataset.txt",2,False)

    # create n gram counts
    test_n_gram.count()

    # export model to text file
    test_n_gram.export_n_gram_model()

    # add witten bell smoothin
    #test_n_gram.witten_bell_smooth()

    test_n_gram.calc_seq_perp("<s> i see as in a map the end of all </s>",2)

    # get test set perplexity
    pp = test_n_gram.calc_perp("testset.txt")
    print(pp)