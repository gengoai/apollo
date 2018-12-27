package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.LibLinear;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Loggable;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

import static com.gengoai.Validation.notNull;

/**
 * A model trained using LibLinear
 *
 * @author David B. Bracewell
 */
public class LibLinearModel extends Classifier implements Loggable {
   private static final long serialVersionUID = 1L;
   private int biasIndex = -1;
   private Model model;

   /**
    * Instantiates a new Lib linear model.
    *
    * @param preprocessors the preprocessors
    */
   public LibLinearModel(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Lib linear model.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public LibLinearModel(Vectorizer<String> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   /**
    * Instantiates a new Lib linear model.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public LibLinearModel(Vectorizer<String> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   /**
    * Instantiates a new Lib linear model.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public LibLinearModel(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   /**
    * Instantiates a new Lib linear model.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public LibLinearModel(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   @Override
   protected Classifier fitPreprocessed(Dataset preprocessed, FitParameters parameters) {
      Parameters fitParameters = notNull(Cast.as(parameters, Parameters.class));
      biasIndex = (fitParameters.bias ? getNumberOfFeatures() + 1 : -1);
      model = LibLinear.fit(() -> preprocessed.stream()
                                              .map(this::encode),
                            new Parameter(fitParameters.solver,
                                          fitParameters.C,
                                          fitParameters.eps,
                                          fitParameters.maxIterations,
                                          fitParameters.p),
                            fitParameters.verbose,
                            getNumberOfFeatures(),
                            biasIndex
                           );
      return this;
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public Classification predict(Example example) {
      return new Classification(LibLinear.estimate(model, encodeAndPreprocess(example), biasIndex),
                                getLabelVectorizer());
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
      /**
       * The Solver to use. (default L2R_LR)
       */
      public SolverType solver = SolverType.L2R_LR;

   }
}//END OF LibLinearModel
