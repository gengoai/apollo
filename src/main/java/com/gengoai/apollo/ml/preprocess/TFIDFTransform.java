package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.stream.MStream;
import com.gengoai.stream.accumulator.MDoubleAccumulator;

import java.util.Optional;

/**
 * <p>Transform values using <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf">Tf-idf</a> </p>
 *
 * @author David B. Bracewell
 */
public class TFIDFTransform extends RestrictedFeaturePreprocessor {
   private static final long serialVersionUID = 1L;
   private volatile Counter<String> documentFrequencies = Counters.newCounter();
   private volatile double totalDocs = 0;


   /**
    * Instantiates a new Tfidf transform.
    */
   public TFIDFTransform() {
      super(null);
   }


   /**
    * Instantiates a new Tfidf transform.
    *
    * @param featureNamePrefix the feature name prefix
    */
   public TFIDFTransform(String featureNamePrefix) {
      super(featureNamePrefix);
   }

   @Override
   public Instance applyInstance(Instance example) {
      final double sum = example.getFeatures()
                                .stream()
                                .filter(this::requiresProcessing)
                                .mapToDouble(Feature::getValue)
                                .sum();
      return example.mapFeatures(in -> {
         if (!requiresProcessing(in)) {
            return Optional.of(in);
         }
         double tfidf = in.value / sum * Math.log(totalDocs / documentFrequencies.get(in.name));
         return Optional.of(Feature.realFeature(in.name, tfidf));
      });
   }

   @Override
   protected void fitFeatures(MStream<Feature> stream) {
      throw new UnsupportedOperationException();
   }

   @Override
   protected void fitInstances(MStream<Example> exampleStream) {
      MDoubleAccumulator docCount = exampleStream.getContext().doubleAccumulator(0d);
      this.documentFrequencies.merge(exampleStream.flatMap(e -> {
         docCount.add(1.0);
         return e.getFeatureNameSpace().distinct();
      }).countByValue());
      this.totalDocs = docCount.value();
   }

   @Override
   public void reset() {
      totalDocs = 0;
      documentFrequencies.clear();
   }

   @Override
   public String toString() {
      return "TFIDFTransform[" + getRestriction() + "]{totalDocuments=" + totalDocs +
                ", vocabSize=" + documentFrequencies.size() + "}";
   }


}//END OF TFIDFTransform
