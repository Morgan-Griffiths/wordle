# import Pkg; Pkg.add("DataStructures")
# import Pkg; Pkg.add("PyCall")
# using Pkg; Pkg.add("ProgressMeter")
using DataStructures
using DelimitedFiles
using Distributed
using ProgressMeter
using Profile

permutations = []
for a in range(1, 3)
    for b in range(1, 3)
        for c in range(1, 3)
            for d in range(1, 3)
                for e in range(1, 3)
                    push!(permutations,(a, b, c, d, e))
                end
            end
        end
    end
end
global result_index_dict = Dict( dist => i for (i,dist) in enumerate(permutations) )
global index_result_dict = Dict( i => dist for (i,dist) in enumerate(permutations) )
# println("permutations = ",length(permutations))

function increment_state(state::Vector,turn::Integer,hidden_word::String,word::String)
    result = evaluate_word(hidden_word,word)
    next_state = update_state(state,turn,result)
    return next_state,result
end

function update_state(state,turn, result)
    # 6, 5, 2
    state[turn, :, :] = result
    return state
end

function evaluate_word(hidden_word::String,word::String)
    MISSING = 1
    CONTAINED = 2
    EXACT = 3
    letter_result = zeros(Int8,5)
    letter_freqs = DataStructures.DefaultDict{Char,Int8}(0)
    for letter in hidden_word
        if letter in keys(letter_freqs)
            letter_freqs[letter] += 1
        else
            letter_freqs[letter] = 1
        end
    end
    for (i,letter) in enumerate(word)
        if letter in keys(letter_freqs) && letter_freqs[letter] > 0
            if letter == hidden_word[i]
                update = EXACT
            elseif letter in hidden_word
                update = CONTAINED
            else
                update = MISSING
            end
        else
            update = MISSING
        end
        letter_result[i] = update
        letter_freqs[letter] = max(0, letter_freqs[letter] - 1)
    end
    return letter_result
end

function filter_words_single_pass(word_arr:: Array{String,2}, guessed_word:: String, results::Vector)
    MISSING = 1
    CONTAINED = 2
    EXACT = 3
    remaining_words = []
    for (i, (letter, result)) in enumerate(zip(guessed_word, results))
        letter = string(letter)
        if result == MISSING
            # remove all words
            a = word_arr[:, 1] .!= letter
            b = word_arr[:, 2] .!= letter
            c = word_arr[:, 3] .!= letter
            d = word_arr[:, 4] .!= letter
            e = word_arr[:, 5] .!= letter
            mask = a .& b .& c .& d .& e
            word_arr = word_arr[mask,:]
        elseif result == EXACT
            mask = word_arr[:, i] .== letter
            word_arr = word_arr[mask,:]
        elseif result == CONTAINED
            numbers = [num for num in range(5) if num != i]
            a = word_arr[:, numbers[1]] .== letter
            b = word_arr[:, numbers[2]] .== letter
            c = word_arr[:, numbers[3]] .== letter
            d = word_arr[:, numbers[4]] .== letter
            mask = a .| b .| c .| d
            word_arr = word_arr[mask,:]
        end
    end
    return remaining_words
end

function filter_words(current_words::Vector,guessed_word::String,results::Vector)
    MISSING = 1
    CONTAINED = 2
    EXACT = 3
    remaining_words = []
    for word in current_words
        for (i,(letter, result)) in enumerate(zip(guessed_word, results))
            # println("result == MISSING is",result == MISSING)
            # println("result == CONTAINED is",result == CONTAINED)
            # println("result == EXACT is",result == EXACT)
            # println("letter ∉ word is",letter ∉ word)
            # println("letter in word is",letter in word)
            if result == MISSING && letter in word
                break
            elseif result == CONTAINED && (letter ∉ word || letter == word[i])
                break
            elseif result == EXACT && letter != word[i]
                break
            elseif i == 4
                push!(remaining_words,word)
            end
        end
    end
    return remaining_words
end

function information(current_words::Integer,remaining_words::Integer)
    remaining_words = max(remaining_words-1,1)
    probability = remaining_words / current_words
    return -log2(probability)
end


function create_first_order_result_distributions(dictionary::Vector{String})
    N = length(dictionary)
    word_result_dist = Dict()
    word_entropys = zeros(Float16,N)
    # starting_information = information(N, 1)
    # println("starting_information = ",starting_information)
    for (i,guess) in enumerate(dictionary)
        # hist_results = []
        # result_matrix = zeros(3 ^ 5)
        entropys = 0
        for hidden_word in dictionary
            result = evaluate_word(hidden_word,guess)
            # println("result",result)
            remaining_words = filter_words(dictionary, guess, result)
            # println("length(remaining_words)",length(remaining_words))
            word_entropy = information(N, length(remaining_words))
            # println("word_entropy",word_entropy)
            # index = result_index_dict[Tuple(result)]
            # result_matrix[index] += 1
            # word_entropy = starting_information - remaining_information
            entropys += word_entropy
        end
        word_entropys[i] = entropys / N
        # word_result_dist[hidden_word] = result_matrix
    end
    (value,index) = findmax(word_entropys)
    return word_entropys, dictionary[index], value
end


function distribution_comparison(dictionary::Array{String,2})
    N = length(dictionary)
    word_result_dist = Dict()
    word_entropys = zeros(Float16,N)
    # starting_information = information(N, 1)
    # println("starting_information = ",starting_information)
    for (i,guess) in enumerate(dictionary)
        # hist_results = []
        # result_matrix = zeros(3 ^ 5)
        entropys = 0
        for hidden_word in dictionary
            result = evaluate_word(hidden_word,guess)
            # println("result",result)
            remaining_words = filter_words_single_pass(dictionary, guess, result)
            # println("length(remaining_words)",length(remaining_words))
            word_entropy = information(N, length(remaining_words))
            # println("word_entropy",word_entropy)
            # index = result_index_dict[Tuple(result)]
            # result_matrix[index] += 1
            # word_entropy = starting_information - remaining_information
            entropys += word_entropy
        end
        word_entropys[i] = entropys / N
        # word_result_dist[hidden_word] = result_matrix
    end
    (value,index) = findmax(word_entropys)
    return word_entropys, dictionary[index], value
end



dictionary = readdlm("data/allowed_words.txt", '\n', String, '\n')
dictionary_nd = Array{String,2}(undef,length(dictionary),5)
for (i,word) in enumerate(dictionary)
    vec = Vector{String}(split(word,"")) 
    for (j,letter) in enumerate(vec)
        dictionary_nd[i,j] = letter
    end
end
# println("dictionary_nd = ",size(dictionary_nd))
# println("dictionary_nd = ",size(dictionary_nd[1:250,:]))
# result = evaluate_word("HELLO","HELMS")
# println("result = $result")
# println("dictionary = ",length(dictionary[1:end]))
# println("dictionary = ",typeof(dictionary[1:100]))

# information test
# entorpy = information(1,5)
# println("entorpy = $entorpy")

# FILTER WORD test
# dictionary = [
#     "MOUNT",
#     "HELLO",
#     "NIXED",
#     "AAHED",
#     "HELMS",
# ]
# result = evaluate_word("AAHED","HELMS")
# println("result = $result")
# @profile filter_words(dictionary[1:500],"HELMS",result)
# Profile.print()
# result = evaluate_word("AAHED","HELMS")
# println("result = $result")
# remaining_words = filter_words(dictionary,"HELMS",result)
# println("remaining_words = $remaining_words")

# DIST
# @time create_first_order_result_distributions(dictionary[1:100])
# @time distribution_comparison(dictionary_nd[1:100,:])
# Profile.print() 
# word_entropy_dist, best_word, most_entropy = create_first_order_result_distributions(dictionary[1:10])
# println("best_word = $best_word")
# println("most_entropy = $most_entropy")
# println("word_entropy_dist = $word_entropy_dist")