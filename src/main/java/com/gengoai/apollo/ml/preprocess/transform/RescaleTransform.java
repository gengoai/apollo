package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.math.Math2;
import com.gengoai.Validation;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.json.JsonReader;
import com.gengoai.json.JsonTokenType;
import com.gengoai.json.JsonWriter;
import com.gengoai.stream.MStream;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class RescaleTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance> {
   private static final String SINGLE_FEATURE = "*****SINGLE_FEATURE*****";
   private double newMin;
   private double newMax;
   private boolean perFeature;
   private Map<String, Double> mins = new HashMap<>();
   private Map<String, Double> maxs = new HashMap<>();

   public RescaleTransform(double newMin, double newMax) {
      this(newMin, newMax, true);
   }

   public RescaleTransform(double newMin, double newMax, boolean perFeature) {
      Validation.checkArgument(newMax > newMin, "max must be > min");
      this.perFeature = perFeature;
      this.newMin = newMin;
      this.newMax = newMax;
   }

   @Override
   public String describe() {
      String name = "RescaleTransform";
      if (!applyToAll()) {
         name += "[" + getRestriction() + "]";
      }
      return name + "{newMin=" + newMin + ", newMax=" + newMax + ", perFeature=" + perFeature + "}";
   }

   @Override
   public void fromJson(@NonNull JsonReader reader) throws IOException {
      reset();
      while (reader.peek() != JsonTokenType.END_OBJECT) {
         switch (reader.peekName()) {
            case "restriction":
               setRestriction(reader.nextKeyValue().v2.asString());
               break;
            case "newMin":
               this.newMin = reader.nextKeyValue().v2.asDoubleValue();
               break;
            case "newMax":
               this.newMax = reader.nextKeyValue().v2.asDoubleValue();
               break;
            case "perFeature":
               this.perFeature = reader.nextKeyValue().v2.asBooleanValue();
               break;
            case "mins":
               this.mins = reader.nextKeyValue().v2.asMap(String.class, Double.class);
               break;
            case "maxs":
               this.maxs = reader.nextKeyValue().v2.asMap(String.class, Double.class);
               break;
         }
      }
   }

   @Override
   public void reset() {
      mins.clear();
      maxs.clear();
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      stream.forEach(features -> {
         for (Feature feature : features) {
            if (perFeature) {
               mins.put(feature.getFeatureName(),
                        FastMath.min(mins.getOrDefault(feature.getFeatureName(), Double.POSITIVE_INFINITY),
                                     feature.getValue()));
               maxs.put(feature.getFeatureName(),
                        FastMath.max(maxs.getOrDefault(feature.getFeatureName(), Double.NEGATIVE_INFINITY),
                                     feature.getValue()));
            } else {
               mins.put(SINGLE_FEATURE, FastMath.min(mins.getOrDefault(SINGLE_FEATURE, Double.POSITIVE_INFINITY),
                                                     feature.getValue()));
               maxs.put(SINGLE_FEATURE, FastMath.max(maxs.getOrDefault(SINGLE_FEATURE, Double.NEGATIVE_INFINITY),
                                                     feature.getValue()));

            }
         }
      });
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.map(feature -> {
         if (perFeature) {
            return Feature.real(feature.getFeatureName(),
                                Math2.rescale(feature.getValue(), mins.get(feature.getFeatureName()),
                                              maxs.get(feature.getFeatureName()), newMin,
                                              newMax));
         }
         return Feature.real(feature.getFeatureName(), Math2.rescale(feature.getValue(), mins.get(SINGLE_FEATURE),
                                                                     maxs.get(SINGLE_FEATURE), newMin, newMax));
      });
   }

   @Override
   public void toJson(@NonNull JsonWriter writer) throws IOException {
      if (!applyToAll()) {
         writer.property("restriction", getRestriction());
      }
      writer.property("newMin", newMin);
      writer.property("newMax", newMax);
      writer.property("perFeature", perFeature);
      writer.property("mins", mins);
      writer.property("maxs", maxs);
   }
}// END OF RescaleTransform
