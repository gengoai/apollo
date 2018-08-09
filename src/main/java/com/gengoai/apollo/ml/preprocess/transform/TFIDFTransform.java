package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.json.JsonEntry;
import com.gengoai.stream.MStream;
import com.gengoai.stream.accumulator.MDoubleAccumulator;
import com.gengoai.string.StringUtils;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * <p>Transform values using <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf">Tf-idf</a> </p>
 *
 * @author David B. Bracewell
 */
public class TFIDFTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private volatile Counter<String> documentFrequencies = Counters.newCounter();
   private volatile double totalDocs = 0;

   /**
    * Instantiates a new Tfidf transform.
    */
   public TFIDFTransform() {
      super(StringUtils.EMPTY);
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
   public String describe() {
      if (applyToAll()) {
         return "TFIDFTransform{totalDocuments=" + totalDocs + ", vocabSize=" + documentFrequencies.size() + "}";
      }
      return "TFIDFTransform[" + getRestriction() + "]{totalDocuments=" + totalDocs + ", vocabSize=" + documentFrequencies
                                                                                                          .size() + "}";
   }

   public static TFIDFTransform fromJson(JsonEntry entry) {
      TFIDFTransform transform = new TFIDFTransform(
         entry.getStringProperty("restriction", null)
      );
      transform.documentFrequencies = Counters.newCounter(entry.getProperty("documentCounts").asMap(Double.class));
      transform.totalDocs = entry.getDoubleProperty("totalDocuments");
      return transform;
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry object = JsonEntry.object();
      if (!applyToAll()) {
         object.addProperty("restriction", getRestriction());
      }
      object.addProperty("documentCounts", documentFrequencies.asMap());
      object.addProperty("totalDocuments", totalDocs);
      return object;
   }

   @Override
   public void reset() {
      totalDocs = 0;
      documentFrequencies.clear();
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      MDoubleAccumulator docCount = stream.getContext().doubleAccumulator(0d);
      this.documentFrequencies.merge(stream.flatMap(instance -> {
                                                       docCount.add(1d);
                                                       return instance.stream().map(Feature::getFeatureName).distinct();
                                                    }
                                                   ).countByValue()
                                    );
      this.totalDocs = docCount.value();
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      double dSum = originalExample.getFeatures().stream().mapToDouble(Feature::getValue).sum();
      return featureStream.map(f -> {
                                  double value = f.getValue() / dSum * Math.log(totalDocs / documentFrequencies.get(f.getFeatureName()));
                                  if (value != 0) {
                                     return Feature.real(f.getFeatureName(), value);
                                  }
                                  return null;
                               }
                              )
                          .filter(Objects::nonNull);
   }


}// END OF TFIDFTransform
