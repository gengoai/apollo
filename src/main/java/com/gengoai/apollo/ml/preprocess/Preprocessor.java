package com.gengoai.apollo.ml.preprocess;


import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;

/**
 * <p>Preprocessors represent filters and transforms to apply to a dataset before building a model. This allows such
 * things as removing bad values or features, feature selection, and value normalization. </p>
 *
 * @author David B. Bracewell
 */
public interface Preprocessor {

   /**
    * Applies the transform tho the given example.
    *
    * @param example the example to apply the transform to
    * @return A new example with the transformed applied
    */
   Example apply(Example example);


   /**
    * Determines the parameters, e.g. counts, etc., of the preprocessor from the given dataset. Implementations should
    * relearn parameters on each call instead of updating.
    *
    * @param dataset the dataset to fit this preprocessors parameters to
    */
   Dataset fitAndTransform(Dataset dataset);


   /**
    * Resets the parameters of the preprocessor.
    */
   void reset();


}//END OF Preprocessor
