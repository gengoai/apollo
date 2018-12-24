package com.gengoai.apollo.ml.pgm;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.collection.counter.Counter;

/**
 * <p>A model to estimates topics for examples.</p>
 *
 * @author David B. Bracewell
 */
public abstract class TopicModel extends Model {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Topic model.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected TopicModel(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   /**
    * Instantiates a new Topic model.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected TopicModel(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   /**
    * Estimates the distribution of topics for the given example
    *
    * @param Example the example to estimate topics for
    * @return An NDArray with topic scores
    */
   public abstract NDArray estimate(Example Example);

   /**
    * Gets the distribution across topics for a given feature.
    *
    * @param feature the feature (word) whose topic distribution is desired
    * @return the distribution across topics for the given feature
    */
   public abstract NDArray getTopicDistribution(String feature);

   /**
    * Gets topic vector.
    *
    * @param topic the topic
    * @return the topic vector
    */
   public abstract NDArray getTopic(int topic);

   /**
    * Gets the words and their probabilities for a given topic
    *
    * @param topic the topic
    * @return the topic words
    */
   public abstract Counter<String> getTopicWords(int topic);


}//END OF TopicModel
