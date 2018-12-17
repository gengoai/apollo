package com.gengoai.apollo.ml.preprocess;


import com.gengoai.Copyable;
import com.gengoai.apollo.ml.Dataset;
import com.gengoai.apollo.ml.Example;

/**
 * <p>Preprocessors represent filters and transforms to apply to a dataset before building a model. This allows such
 * things as removing bad values or features, feature selection, and value normalization. Preprocessors can be
 * build-time only (i.e. trainOnly is true) or can be required when the model is applied, e.g. value normalization
 * transforms. Implementations should not reuse examples in the apply function.</p>
 *
 * @author David B. Bracewell
 */
public interface Preprocessor extends Copyable<Preprocessor> {

   /**
    * Applies the transform tho the given example.
    *
    * @param example the example to apply the transform to
    * @return A new example with the transformed applied
    */
   Example apply(Example example);

   /**
    * Provides a simple description of the preprocessor and its parameters, if any.
    *
    * @return a description string
    */
   String describe();

   /**
    * Determines the parameters, e.g. counts, etc., of the preprocessor from the given dataset
    *
    * @param dataset the dataset to fit this preprocessors parameters to
    */
   void fit(Dataset dataset);


   /**
    * Determines if the fit method is required to be called for the preprocessor to function correctly.
    *
    * @return True if the fit method must be called before applying, False otherwise
    */
   default boolean requiresFit() {
      return true;
   }

   /**
    * Resets the parameters of the preprocessor.
    */
   void reset();

   /**
    * Determines if the preprocessor only needs to be applied when building a model
    *
    * @return True the preprocessor is only required at model train time, False the preprocessor must be applied when
    * the model is used as well.
    */
   boolean trainOnly();


}//END OF Preprocessor
