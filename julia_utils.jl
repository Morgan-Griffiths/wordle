# import Pkg; Pkg.add("DataStructures")
# import Pkg; Pkg.add("PyCall")
# using Pkg; Pkg.add("ProgressMeter")
using DataStructures
using DelimitedFiles
using Distributed
using ProgressMeter

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
    letter_result = zeros(5)
    letter_freqs = DataStructures.DefaultDict(0)
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


function filter_words(current_words::Vector,guessed_word::String,results::Vector)
    EXACT = 3.0
    CONTAINED = 2.0
    MISSING = 1.0
    remaining_words = []
    for word in current_words
        for (i, (letter, result)) in enumerate(zip(guessed_word, results))
            # println("result == MISSING is",result == MISSING)
            # println("result == CONTAINED is",result == CONTAINED)
            # println("result == EXACT is",result == EXACT)
            # println("letter ∉ word is",letter ∉ word)
            # println("letter in word is",letter in word)
            if result == MISSING && letter in word
                break
            elseif result == EXACT && letter != word[i]
                break
            elseif result == CONTAINED && letter ∉ word
                break
            elseif i == 4
                push!(remaining_words,word)
            end
        end
    end
    return remaining_words
end

function information(current_words::Integer,remaining_words::Integer)
    if remaining_words == 0
        return 0
    end
    probability = remaining_words / current_words
    return -log2(probability)
end


function create_first_order_result_distributions(dictionary::Vector{String})
    word_result_dist = Dict()
    word_entropy_dist = DataStructures.DefaultDict([])
    starting_information = information(length(dictionary), 1)
    println("starting_information = ",starting_information)
    @showprogress for word in dictionary
        hidden_word = word
        # hist_results = []
        result_matrix = zeros(3 ^ 5)
        for guess in dictionary
            result = evaluate_word(hidden_word,guess)
            remaining_words = filter_words(dictionary, guess, result)
            remaining_information = information(length(dictionary), length(remaining_words))
            index = result_index_dict[Tuple(result)]
            # println("remaining_information = ",remaining_information)
            result_matrix[index] += 1
            word_entropy = starting_information - remaining_information
            # println("word_entropy = ",word_entropy)
            push!(word_entropy_dist[guess],word_entropy)
        end
        word_result_dist[word] = result_matrix
    end
    best_word = ""
    most_entropy = 0
    println("Crunching entropies")
    @showprogress for word in dictionary
        word_entropy_dist[word] = sum(word_entropy_dist[word]) / length(word_entropy_dist[word])
        if word_entropy_dist[word] > most_entropy
            best_word = word
            most_entropy = word_entropy_dist[word]
        end
    end
    return word_result_dist, word_entropy_dist, best_word, most_entropy
end


dictionary = readdlm("wordle.txt", '\n', String, '\n')
# println("dictionary = ",typeof(dictionary))
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
# remaining_words = filter_words(dictionary,"HELMS",result)
# println("remaining_words = $remaining_words")

# DIST
word_result_dist, word_entropy_dist, best_word, most_entropy = create_first_order_result_distributions(dictionary[1:500])
println("best_word = $best_word")
println("most_entropy = $most_entropy")