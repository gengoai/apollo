package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.stream.MStream;
import com.gengoai.string.Strings;

import java.io.Serializable;

/**
 * <p>A preprocessor that is only applied to a restricted set of features based on the prefix of the feature
 * name, e.g. if the restriction was <code>WORD</code>, only features beginning with the prefix <code>WORD=</code> would
 * have the preprocessor applied. Note the preprocessor will add <code>=</code> at the end of the prefix if not
 * specified. All features can be processed by providing an empty or null prefix.</p>
 *
 * @author David B. Bracewell
 */
public abstract class RestrictedFeaturePreprocessor implements InstancePreprocessor, Serializable {
   private static final long serialVersionUID = 1L;
   private final String featureNamePrefix;
   private final boolean acceptAll;

   /**
    * Instantiates a new Restricted feature preprocessor.
    *
    * @param featureNamePrefix the feature name prefix
    */
   protected RestrictedFeaturePreprocessor(String featureNamePrefix) {
      this.featureNamePrefix = Strings.appendIfNotPresent(featureNamePrefix, "=");
      this.acceptAll = Strings.isNullOrBlank(featureNamePrefix);
   }


   @Override
   public final Dataset fitAndTransform(Dataset dataset) {
      reset();
      fitInstances(dataset.stream().flatMap(Example::stream));
      Dataset out = dataset.mapSelf(this::apply);
      cleanup();
      return out;
   }

   /**
    * Gets a string representation of  restriction. In the case that no restriction is in place it will return
    * <code>*</code>
    *
    * @return the restriction
    */
   public final String getRestriction() {
      return Strings.isNullOrBlank(featureNamePrefix) ? "*" : featureNamePrefix;
   }


   /**
    * Cleanup.
    */
   protected void cleanup() {

   }

   /**
    * Fit implementation working over individual examples in the dataset. Default implementation flattens out the
    * examples, i.e. sequences are split into individual components.
    *
    * @param exampleStream the example stream
    */
   protected void fitInstances(MStream<Example> exampleStream) {
      fitFeatures(exampleStream.flatMap(e -> e.getFeatures().stream())
                               .filter(this::requiresProcessing));
   }

   /**
    * Fit implementation working over individual features in the dataset
    *
    * @param exampleStream the example stream
    */
   protected abstract void fitFeatures(MStream<Feature> exampleStream);


   /**
    * Requires processing boolean.
    *
    * @param f the f
    * @return the boolean
    */
   public boolean requiresProcessing(Feature f) {
      return acceptAll || f.hasPrefix(featureNamePrefix);
   }

}//END OF RestrictedFeaturePreprocessor
