package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * <p>An instance preprocessor that is only applied to a restricted set of features based on the prefix of the feature
 * name, e.g. if the restriction was <code>WORD=</code>, only features beginning with the prefix <code>WORD=</code>
 * would have the preprocessor applied.</p></p>
 *
 * @author David B. Bracewell
 */
public abstract class RestrictedInstancePreprocessor implements InstancePreprocessor, Serializable {
   private static final long serialVersionUID = 1L;
   private String featureNamePrefix;
   private boolean acceptAll;

   /**
    * Instantiates a new Restricted filter. Null or blank strings cause the preprocessor to be applied to all features.
    *
    * @param featureNamePrefix the feature name prefix
    */
   protected RestrictedInstancePreprocessor(String featureNamePrefix) {
      this.featureNamePrefix = featureNamePrefix;
      this.acceptAll = StringUtils.isNullOrBlank(featureNamePrefix);
   }

   /**
    * Instantiates a new Restricted instance preprocessor.
    */
   protected RestrictedInstancePreprocessor() {
      this.featureNamePrefix = null;
      this.acceptAll = true;
   }

   @Override
   public final Instance apply(Instance example) {
      if (applyToAll()) {
         return Instance.create(restrictedProcessImpl(example.stream(), example).collect(Collectors.toList()),
                                example.getLabel());
      }
      return Instance.create(Stream.concat(restrictedProcessImpl(shouldFilter(example), example),
                                           shouldNotFilter(example)).collect(Collectors.toList()),
                             example.getLabel()
                            );
   }

   /**
    * Checks if the preprocessor should be applied to all features, i.e. no restriction
    *
    * @return True apply to all features, False apply the restriction
    */
   public boolean applyToAll() {
      return acceptAll;
   }

   @Override
   public final void fit(Dataset<Instance> dataset) {
      if (requiresFit()) {
         restrictedFitImpl(dataset.stream().map(i -> shouldFilter(i).collect(Collectors.toList())));
      }
   }

   /**
    * Gets the restriction.
    *
    * @return the restriction
    */
   public String getRestriction() {
      return featureNamePrefix;
   }

   /**
    * Sets the restriction.
    *
    * @param prefix the feature name prefix
    */
   protected final void setRestriction(String prefix) {
      this.featureNamePrefix = prefix;
      this.acceptAll = StringUtils.isNullOrBlank(featureNamePrefix);
   }

   /**
    * Implementation of the preprocessors fit only against features that should be restricted.
    *
    * @param stream the stream of features
    */
   protected abstract void restrictedFitImpl(MStream<List<Feature>> stream);

   /**
    * Restricted process stream.
    *
    * @param featureStream   the feature stream
    * @param originalExample the original example
    * @return the stream
    */
   protected abstract Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample);

   /**
    * Gets the features from the given example that  match the restriction.
    *
    * @param example the example
    * @return the stream of features that match the restriction
    */
   private Stream<Feature> shouldFilter(Instance example) {
      return example.getFeatures().stream()
                    .filter(
                       f -> !f.getName().equals("SPECIAL::BIAS_FEATURE") || (applyToAll() || f.getName().startsWith(
                          getRestriction())));
   }

   /**
    * Gets the features from the given example that do not match the restriction.
    *
    * @param example the example
    * @return the stream of features that do not match the restriction
    */
   private Stream<Feature> shouldNotFilter(@NonNull Instance example) {
      return example.getFeatures().stream()
                    .filter(
                       f -> f.getName().equals("SPECIAL::BIAS_FEATURE") || (!applyToAll() && !f.getName().startsWith(
                          getRestriction())));
   }

   @Override
   public String toString() {
      return describe();
   }

}// END OF RestrictedInstancePreprocessor
