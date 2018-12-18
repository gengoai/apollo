package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.*;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.DoubleVectorizer;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

/**
 * <p>Wraps a {@link Regression} allowing it to work directly with {@link Dataset}s and {@link Example}s instead
 * of NDArray</p>
 *
 * @author David B. Bracewell
 */
public class PipelinedRegression extends PipelinedModel implements Regression {
   private static final long serialVersionUID = 1L;
   private final Regression regression;

   /**
    * Instantiates a new Pipelined model.
    *
    * @param regression        the regression
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public PipelinedRegression(Regression regression, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(new DoubleVectorizer(), featureVectorizer, preprocessors);
      this.regression = regression;
   }

   /**
    * Instantiates a new Pipelined regression.
    *
    * @param regression    the regression
    * @param preprocessors the preprocessors
    */
   public PipelinedRegression(Regression regression, PreprocessorList preprocessors) {
      super(new DoubleVectorizer(), IndexVectorizer.featureVectorizer(), preprocessors);
      this.regression = regression;
   }

   @Override
   public NDArray estimate(NDArray data) {
      return regression.estimate(data);
   }

   /**
    * Estimates a real-value based on the input instance.
    *
    * @param vector the instance
    * @return the estimated value
    */
   public double estimateScalar(Example vector) {
      return estimate(encodeAndPreprocess(vector)).scalarValue();
   }

   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters fitParameters) {
      regression.fit(dataSupplier, fitParameters);
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return regression.getDefaultFitParameters();
   }

   @Override
   public int getNumberOfFeatures() {
      return getFeatureVectorizer().size();
   }

   @Override
   public int getNumberOfLabels() {
      return 0;
   }

}//END OF PipelinedRegression
