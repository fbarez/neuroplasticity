def main():
    print("Running analysis experiment...")
    model_trainer = train_model()
    retrained_analyser = retrain_pruned(model_trainer)
    basic_analyser = analyse_model(basic_model_path, basic_activations_path)

    # Perform analysis
    new_concept_neurons = retrained_analyser.identify_concept_neurons()
    post_top_words = retrained_analyser.show_top_words(new_concept_neurons)
    pre_top_words = basic_analyser.show_top_words(new_concept_neurons)
    print(post_top_words)
    print(pre_top_words)
    plastic_df = pd.DataFrame.from_dict(pre_top_words, orient='index', columns=['pre_top_words'])
    plastic_df['post_top_words'] = post_top_words.values()
    plastic_df.to_csv('data/processed/locations_plastic.csv')

if __name__ == '__main__':
    main()