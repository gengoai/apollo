package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.DoubleVectorizer;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;

/**
 * <p>Base regression model that produces a real-value for an input instance.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Regression extends Model<Regression> {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Regression.
    */
   public Regression(Preprocessor... preprocessors) {
      super(new DoubleVectorizer(), IndexVectorizer.featureVectorizer(), preprocessors);
   }

   /**
    * Instantiates a new Regression.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public Regression(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(new DoubleVectorizer(), featureVectorizer, preprocessors);
   }

   /**
    * Instantiates a new Regression.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public Regression(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(new DoubleVectorizer(), featureVectorizer, preprocessors);
   }

   /**
    * Estimates a real-value based on the input instance.
    *
    * @param vector the instance
    * @return the estimated value
    */
   public abstract double estimate(Example vector);


}//END OF Regression
