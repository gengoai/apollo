package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;

/**
 * <p>Labels each example in a sequence of examples, which may represent points in time, tokens in a sentence, etc.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class SequenceLabeler extends Model {
   private static final long serialVersionUID = 1L;


   public SequenceLabeler(Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(),
            IndexVectorizer.featureVectorizer(),
            preprocessors);
   }

   public SequenceLabeler(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public SequenceLabeler(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   public SequenceLabeler(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public SequenceLabeler(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   /**
    * Specialized transform to predict the labels for a sequence.
    *
    * @param example the example sequence to label
    */
   public abstract Labeling label(Example example);


   @Override
   @SuppressWarnings("unchecked")
   public Vectorizer<String> getLabelVectorizer() {
      return Cast.as(super.getLabelVectorizer());
   }

}//END OF SequenceLabeler
