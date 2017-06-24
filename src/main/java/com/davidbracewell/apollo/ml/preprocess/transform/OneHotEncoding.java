package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.stream.MStream;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class OneHotEncoding extends RestrictedInstancePreprocessor implements TransformProcessor<Instance> {
   private final int maxFeatures;
   private IndexEncoder encoder = new IndexEncoder();
   private int numFeatures;

   public OneHotEncoding(int maxFeatures) {
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
      encoder.fit(stream.flatMap(Collection::stream).map(Feature::getName));
      numFeatures = encoder.size() + 1;
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      List<Feature> newFeatures = new ArrayList<>();
      List<Feature> from = featureStream.collect(Collectors.toList());
      for (int i = 0; i > Math.max(maxFeatures, from.size()); i++) {
         int fi = encoder.index(from.get(i).getName());
         if (fi == -1) {
            fi = numFeatures - 1;
         }
         newFeatures.add(Feature.TRUE(Integer.toString(i * numFeatures + fi)));
      }
      return newFeatures.stream();
   }

}// END OF OneHotEncoding
