package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.*;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

/**
 * <p>Wraps a {@link SequenceLabeler} allowing it to work directly with {@link Dataset}s and {@link Example}s instead
 * of NDArray</p>
 *
 * @author David B. Bracewell
 */
public class PipelinedSequenceLabeler extends PipelinedModel implements SequenceLabeler {
   private static final long serialVersionUID = 1L;
   private final SequenceLabeler sequenceLabeler;

   /**
    * Instantiates a new Pipelined sequence labeler.
    *
    * @param sequenceLabeler the sequence labeler
    * @param preprocessors   the preprocessors
    */
   public PipelinedSequenceLabeler(SequenceLabeler sequenceLabeler,
                                   Preprocessor... preprocessors
                                  ) {
      this(sequenceLabeler, IndexVectorizer.featureVectorizer(), new PreprocessorList(preprocessors));
   }

   /**
    * Instantiates a new Pipelined sequence labeler.
    *
    * @param sequenceLabeler   the sequence labeler
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public PipelinedSequenceLabeler(SequenceLabeler sequenceLabeler,
                                   Vectorizer<String> featureVectorizer,
                                   PreprocessorList preprocessors
                                  ) {
      super(new IndexVectorizer(true), featureVectorizer, preprocessors);
      this.sequenceLabeler = sequenceLabeler;
   }

   @Override
   public NDArray estimate(NDArray data) {
      return sequenceLabeler.estimate(data);
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
      return new Labeling(estimate(data).getPredictedAsNDArray(), Cast.as(labelVectorizer));
   }

   /**
    * Specialized transform to predict the labels for a sequence.
    *
    * @param example the example sequence to label
    */
   public Labeling label(Example example) {
      return label(encodeAndPreprocess(example));
   }

}//END OF PipelinedClassifier
