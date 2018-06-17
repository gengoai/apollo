package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.conversion.Val;
import com.gengoai.json.JsonReader;
import com.gengoai.json.JsonTokenType;
import com.gengoai.json.JsonWriter;
import com.gengoai.stream.MStream;
import com.gengoai.stream.accumulator.MDoubleAccumulator;
import com.gengoai.string.StringUtils;
import com.gengoai.tuple.Tuple2;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.Map;
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

   @Override
   public void fromJson(@NonNull JsonReader reader) throws IOException {
      reset();
      while (reader.peek() != JsonTokenType.END_OBJECT) {
         switch (reader.peekName()) {
            case "restriction":
               setRestriction(reader.nextKeyValue().v2.asString());
               break;
            case "totalDocuments":
               this.totalDocs = reader.nextKeyValue().v2.asDoubleValue();
               break;
            case "documentCounts":
               reader.beginObject();
               while (reader.peek() != JsonTokenType.END_OBJECT) {
                  Tuple2<String, Val> kv = reader.nextKeyValue();
                  documentFrequencies.set(kv.getKey(), kv.getValue().asDoubleValue());
               }
               reader.endObject();
               break;
         }
      }
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

   @Override
   public void toJson(@NonNull JsonWriter writer) throws IOException {
      if (!applyToAll()) {
         writer.property("restriction", getRestriction());
      }
      writer.property("totalDocuments", totalDocs);
      writer.beginObject("documentCounts");
      for (Map.Entry<String, Double> entry : documentFrequencies.entries()) {
         writer.property(entry.getKey(), entry.getValue());
      }
      writer.endObject();
   }
}// END OF TFIDFTransform
