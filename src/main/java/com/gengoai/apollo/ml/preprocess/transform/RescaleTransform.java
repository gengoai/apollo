package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.json.JsonEntry;
import com.gengoai.math.Math2;
import com.gengoai.stream.MStream;
import org.apache.commons.math3.util.FastMath;

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

   public static RescaleTransform fromJson(JsonEntry entry) {
      RescaleTransform transform = new RescaleTransform(
         entry.getDoubleProperty("newMin"),
         entry.getDoubleProperty("newMax"),
         entry.getBooleanProperty("perFeature", false)
      );
      transform.setRestriction(entry.getStringProperty("restriction", null));
      transform.mins = entry.getProperty("mins").getAsMap(Double.class);
      transform.maxs = entry.getProperty("maxs").getAsMap(Double.class);
      return transform;
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry object = JsonEntry.object();
      if (!applyToAll()) {
         object.addProperty("restriction", getRestriction());
      }
      object.addProperty("newMin", newMin);
      object.addProperty("newMax", newMax);
      object.addProperty("perFeature", perFeature);
      object.addProperty("mins", mins);
      object.addProperty("maxs", maxs);
      return object;
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


}// END OF RescaleTransform
