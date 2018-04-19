package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.encoder.IndexEncoder;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.mango.stream.MStream;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class OneHotEncoding extends RestrictedInstancePreprocessor implements TransformProcessor<Instance> {
   private static final long serialVersionUID = 1L;
   private final int maxFeatures;
   private IndexEncoder encoder = new IndexEncoder();
   private int numFeatures;

   public OneHotEncoding(String featureNamePrefix) {
      super(featureNamePrefix);
      this.maxFeatures = Integer.MAX_VALUE - 1;
   }

   public OneHotEncoding(String featureNamePrefix, int maxFeatures) {
      super(featureNamePrefix);
      this.maxFeatures = maxFeatures;
   }

   public OneHotEncoding(int maxFeatures) {
      super();
      this.maxFeatures = maxFeatures;
   }

   public OneHotEncoding() {
      this(Integer.MAX_VALUE - 1);
   }

   @Override
   public String describe() {
      return "OneHotEncoding";
   }

   @Override
   public void reset() {
      encoder = new IndexEncoder();
      numFeatures = 0;
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      encoder.unFreeze();
      encoder.fit(stream.flatMap(Collection::stream).map(Feature::getFeatureName));
      encoder.freeze();
      numFeatures = encoder.size() + 1;
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      List<Feature> newFeatures = new ArrayList<>();
      List<Feature> from = featureStream.collect(Collectors.toList());
      for (int i = 0; i < Math.min(maxFeatures, from.size()); i++) {
         int fi = encoder.index(from.get(i).getFeatureName());
         if (fi == -1) {
            fi = numFeatures - 1;
         }
         newFeatures.add(Feature.TRUE(Integer.toString(i * numFeatures + fi)));
      }
      return newFeatures.stream();
   }

}// END OF OneHotEncoding
