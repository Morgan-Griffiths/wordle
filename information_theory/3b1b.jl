# import Pkg; Pkg.add("JSON")
# import Pkg; Pkg.add("IterTools")
# import Pkg; Pkg.add("Npy")
using DelimitedFiles
using IterTools
import JSON
# using Npy: NpyArray, sync!
using NPZ

# global MISSING::UInt8 = 0
# global CONTAINED::UInt8 = 1
# global EXACT::UInt8 = 2

global DATA_DIR = joinpath((pwd(),"data"))
global PATTERN_GRID_DATA = Dict()
global PATTERN_MATRIX_FILE = joinpath(DATA_DIR, "pattern_matrix.npy")
global SHORT_WORD_LIST_FILE = joinpath(DATA_DIR, "possible_words.txt")
global LONG_WORD_LIST_FILE = joinpath(DATA_DIR, "allowed_words.txt")
global WORD_FREQ_FILE = joinpath(DATA_DIR, "wordle_words_freqs_full.txt")
global WORD_FREQ_MAP_FILE = joinpath(DATA_DIR, "freq_map.json")

function get_word_list(;short=false)
    if short
        file = SHORT_WORD_LIST_FILE  
    else 
        file = LONG_WORD_LIST_FILE
    end
    result = readdlm(file, '\n', String, '\n')
    return result
end


function sigmoid(x)
    return 1 / 1 + ℯ ^ (-x)
end

function get_word_frequencies(;regenerate::Bool=false)
    if isfile(WORD_FREQ_MAP_FILE) || regenerate
        result = JSON.parsefile(WORD_FREQ_MAP_FILE)
        return result
    end
    # Otherwise, regenerate
    freq_map = Dict()
    open(WORD_FREQ_FILE,"r") do fp
        for line in readlines(fp)
            pieces = line.split(" ")
            word = pieces[0]
            freqs = [float(piece.strip()) for piece in pieces[2:end]]
            freq_map[word] = mean(freqs[-5:end])
        end
    end
    open(WORD_FREQ_MAP_FILE, "w") do fp
        string_data = JSON.json(freq_map)
        write(fp,string_data)
    end
    return freq_map
end


function get_frequency_based_priors(;n_common=3000, width_under_sigmoid=10)
    """
    We know that that list of wordle answers was curated by some human
    based on whether they're sufficiently common. This function aims
    to associate each word with the likelihood that it would actually
    be selected for the final answer.
    Sort the words by frequency, then apply a sigmoid along it.
    """
    freq_map = get_word_frequencies()
    words = collect(keys(freq_map))
    freqs = Array([freq_map[w] for w in words])
    arg_sort = sortperm(freqs)
    sorted_words = words[arg_sort]

    # We want to imagine taking this sorted list, and putting it on a number
    # line so that it's length is 10, situating it so that the n_common most common
    # words are positive, then applying a sigmoid
    x_width = width_under_sigmoid
    c = x_width * (-0.5 + n_common / length(words))
    xs = LinRange(c - x_width / 2, c + x_width / 2, length(words))
    priors = Dict()
    for (word, x) in zip(sorted_words, xs)
        priors[word] = sigmoid(x)
    end
    return priors
end


function get_weights(words, priors)
    frequencies = Array([priors[word] for word in words])
    total = sum(frequencies)
    if total == 0
        return zeros(size(frequencies))
    end
    return frequencies / total
end

function words_to_int_arrays(words)
    return Array{UInt8}([[Int(c) for c in w] for w in words])
end


function get_pattern_matrix(words1, words2)
    if isempty(PATTERN_GRID_DATA)
        # PATTERN_GRID_DATA["grid"] = NpyArray(PATTERN_MATRIX_FILE)
        PATTERN_GRID_DATA["grid"] = npzread(PATTERN_MATRIX_FILE)
        PATTERN_GRID_DATA["words_to_index"] = Dict(word => i for (i,word) in enumerate(get_word_list()))
    end
    full_grid = PATTERN_GRID_DATA["grid"]
    words_to_index = PATTERN_GRID_DATA["words_to_index"]

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]
    mesh = ix(indices1, indices2)
    return full_grid[mesh.x,mesh.y]
end

function get_pattern_distributions(allowed_words:: Matrix, possible_words:: Matrix, weights:: Matrix)
    """
    For each possible guess in allowed_words, this finds the probability
    distribution across all of the 3^5 wordle patterns you could see, assuming
    the possible answers are in possible_words with associated probabilities
    in weights.
    It considers the pattern hash grid between the two lists of words, and uses
    that to bucket together words from possible_words which would produce
    the same pattern, adding together their corresponding probabilities.
    """
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)
    n = length(allowed_words)
    distributions = zeros((n, 3 ^ 5))
    n_range = hcat(1:n)
    for (j, prob) in enumerate(weights)
        distributions[n_range, pattern_matrix[:, j]] += prob
    end
    return distributions
end

function ix(xin,yin)
    xout = reshape(xin,(size(xin)...,1))
    yout = reshape(yin,(1,size(yin)...))
    return (x=xout, y=yout)
end

function entropy_of_distributions(distributions;, atol=1e-12)
    axis = length(distributions.shape) - 1
    return entropy(distributions, base=2, axis=axis)
end


function get_entropies(allowed_words, possible_words, weights)
    if sum(weights) == 0
        return zeros(length(allowed_words))
    end
    distributions = get_pattern_distributions(allowed_words, possible_words, weights)
    return entropy_of_distributions(distributions)
end

words = get_word_list()
priors = get_frequency_based_priors()
weights = get_weights(words,priors)
# words = [
#     "MOUNT",
#     "HELLO",
#     "NIXED",
#     "AAHED",
#     "HELMS",
# ]
# target = [[242   3  27   0   1]
#  [ 81 242   3   4  26]
#  [  1  27 242 216  27]
#  [  0  36 216 242  36]
#  [ 27  26   3   4 242]]
dist = get_pattern_distributions(words, words,weights)
println(dist)