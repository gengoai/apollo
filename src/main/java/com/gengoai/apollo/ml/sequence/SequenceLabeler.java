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
   private final Validator sequenceValidator;


   public SequenceLabeler(Validator validator, Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(),
            IndexVectorizer.featureVectorizer(),
            preprocessors);
      this.sequenceValidator = validator;
   }


   public SequenceLabeler(Validator sequenceValidator, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
      this.sequenceValidator = sequenceValidator;
   }

   public SequenceLabeler(Validator sequenceValidator, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
      this.sequenceValidator = sequenceValidator;
   }

   /**
    * Specialized transform to predict the labels for a sequence.
    *
    * @param example the example sequence to label
    */
   public abstract Labeling label(Example example);


   protected boolean isValidTransition(int current, int previous, Example example) {
      return sequenceValidator.isValid(getLabelVectorizer().decode(current),
                                       getLabelVectorizer().decode(previous),
                                       example);
   }

   protected boolean isValidTransition(String current, String previous, Example example) {
      return sequenceValidator.isValid(current, previous, example);
   }

   @Override
   @SuppressWarnings("unchecked")
   public Vectorizer<String> getLabelVectorizer() {
      return Cast.as(super.getLabelVectorizer());
   }

}//END OF SequenceLabeler
