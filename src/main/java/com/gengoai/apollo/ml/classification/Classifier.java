package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;

/**
 * Base class for classifiers that predicts the label, or class, for a set of features.
 *
 * @author David B. Bracewell
 */
public abstract class Classifier extends Model<Classifier> {
   private static final long serialVersionUID = 1L;

   public Classifier(Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(),
            IndexVectorizer.featureVectorizer(),
            preprocessors);
   }

   /**
    * Instantiates a new Classifier.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected Classifier(Vectorizer<String> labelVectorizer,
                        Vectorizer<String> featureVectorizer,
                        Preprocessor... preprocessors
                       ) {
      super(labelVectorizer,
            featureVectorizer,
            new PreprocessorList(preprocessors));
   }

   /**
    * Instantiates a new Classifier.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected Classifier(Vectorizer<String> labelVectorizer,
                        Vectorizer<String> featureVectorizer,
                        PreprocessorList preprocessors
                       ) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }


   /**
    * Instantiates a new Classifier.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected Classifier(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(),
            featureVectorizer,
            new PreprocessorList(preprocessors));
   }

   /**
    * Instantiates a new Classifier.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected Classifier(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(IndexVectorizer.labelVectorizer(),
            featureVectorizer,
            preprocessors);
   }

   @Override
   @SuppressWarnings("unchecked")
   public Vectorizer<String> getLabelVectorizer() {
      return Cast.as(super.getLabelVectorizer());
   }

   /**
    * <p>Predicts the label(s) for a given example encoding and preprocessing the example as needed.</p>
    *
    * @param example the example
    * @return the classification result
    */
   public abstract Classification predict(Example example);


}//END OF Classifier
