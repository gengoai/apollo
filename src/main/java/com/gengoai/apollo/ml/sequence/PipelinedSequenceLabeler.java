package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.*;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

/**
 * @author David B. Bracewell
 */
public class PipelinedSequenceLabeler extends PipelinedModel implements SequenceLabeler {
   private static final long serialVersionUID = 1L;
   private final SequenceLabeler sequenceLabeler;

   public PipelinedSequenceLabeler(SequenceLabeler sequenceLabeler,
                                   Preprocessor... preprocessors
                                  ) {
      this(sequenceLabeler, IndexVectorizer.featureVectorizer(), new PreprocessorList(preprocessors));
   }

   public PipelinedSequenceLabeler(SequenceLabeler sequenceLabeler,
                                   Vectorizer<String> featureVectorizer,
                                   PreprocessorList preprocessors
                                  ) {
      super(new IndexVectorizer(true), featureVectorizer, preprocessors);
      this.sequenceLabeler = sequenceLabeler;
   }

   @Override
   public SequenceLabeler copy() {
      PipelinedSequenceLabeler copy = new PipelinedSequenceLabeler(sequenceLabeler.copy(),
                                                                   Cast.as(featureVectorizer),
                                                                   preprocessors.copy());
      preprocessors.forEach(p -> copy.preprocessors.add(p.copy()));
      return copy;
   }

   @Override
   public NDArray estimate(NDArray data) {
      return sequenceLabeler.estimate(data);
   }

   @Override
   public Evaluation evaluate(Dataset evaluationData) {
      return null;
   }


   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters fitParameters) {
      sequenceLabeler.fit(dataSupplier, fitParameters);
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return sequenceLabeler.getDefaultFitParameters();
   }

   @Override
   public int getNumberOfFeatures() {
      return sequenceLabeler.getNumberOfFeatures();
   }

   @Override
   public int getNumberOfLabels() {
      return sequenceLabeler.getNumberOfLabels();
   }

   @Override
   public Labeling label(NDArray data) {
      return new Labeling(estimate(data), Cast.as(labelVectorizer));
   }

   public Labeling label(Example example) {
      return label(featureVectorizer.transform(example));
   }

}//END OF PipelinedClassifier
