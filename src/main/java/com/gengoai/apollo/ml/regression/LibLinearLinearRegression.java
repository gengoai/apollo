package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.LibLinear;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

/**
 * <p>Regression model using LibLinear</p>
 *
 * @author David B. Bracewell
 */
public class LibLinearLinearRegression extends Regression {
   private static final long serialVersionUID = 1L;
   private Model model;
   private int biasIndex;


   /**
    * Instantiates a new Lib linear linear regression.
    */
   public LibLinearLinearRegression() {
   }

   /**
    * Instantiates a new Lib linear linear regression.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public LibLinearLinearRegression(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   public LibLinearLinearRegression(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   @Override
   public double estimate(Example data) {
      return LibLinear.regress(model, encodeAndPreprocess(data), biasIndex);
   }

   private void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      biasIndex = (fitParameters.bias ? getNumberOfFeatures() + 1 : -1);
      model = LibLinear.fit(dataSupplier,
                            new Parameter(SolverType.L2R_L2LOSS_SVR,
                                          fitParameters.C,
                                          fitParameters.eps,
                                          fitParameters.maxIterations,
                                          fitParameters.p),
                            fitParameters.verbose,
                            getNumberOfFeatures(),
                            biasIndex
                           );
   }

   @Override
   public void fitPreprocessed(Dataset dataSupplier, FitParameters fitParameters) {
      fit(() -> dataSupplier.stream().map(this::encode), Cast.as(fitParameters, Parameters.class));
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return new Parameters();
   }

   /**
    * Custom fit parameters for LibLinear
    */
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The cost parameter (default 1.0)
       */
      public double C = 1.0;
      /**
       * Use a bias feature or not. (default false)
       */
      public boolean bias = false;
      /**
       * The tolerance for termination.(default 0.0001)
       */
      public double eps = 0.0001;
      /**
       * The maximum number of iterations to run the trainer (Default 1000)
       */
      public int maxIterations = 1000;

      /**
       * The epsilon in loss function of epsilon-SVR (default 0.1)
       */
      public double p = 0.1;

   }

}//END OF MultivariateLinearRegression
