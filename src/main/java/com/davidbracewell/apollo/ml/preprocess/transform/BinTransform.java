package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.EnhancedDoubleStatistics;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.list.PrimitiveArrayList;
import com.davidbracewell.conversion.Val;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MStatisticsAccumulator;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * <p>Converts a real valued feature into a number of binary features by creating a <code>bin</code> number of new
 * binary features. Creates <code>bin</code> number of equal sized bins that the feature values can fall into.</p>
 *
 * @author David B. Bracewell
 */
public class BinTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private double[] bins;


   /**
    * Instantiates a new BinTransform with no restriction
    *
    * @param numberOfBins the number of bins to convert the feature into
    */
   public BinTransform(int numberOfBins) {
      Preconditions.checkArgument(numberOfBins > 0, "Number of bins must be > 0.");
      this.bins = new double[numberOfBins];
   }

   /**
    * Instantiates a new BinTransform.
    *
    * @param featureNamePrefix the feature name prefix to restrict to
    * @param numberOfBins      the number of bins to convert the feature into
    */
   public BinTransform(@NonNull String featureNamePrefix, int numberOfBins) {
      super(featureNamePrefix);
      Preconditions.checkArgument(numberOfBins > 0, "Number of bins must be > 0.");
      this.bins = new double[numberOfBins];
   }

   /**
    * Instantiates a new Real to discrete transform.
    */
   protected BinTransform() {
      this(StringUtils.EMPTY, 1);
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      MStatisticsAccumulator stats = stream.getContext().statisticsAccumulator();
      stream.forEach(instance -> stats.combine(instance.stream()
                                                       .mapToDouble(Feature::getValue)
                                                       .collect(EnhancedDoubleStatistics::new,
                                                                EnhancedDoubleStatistics::accept,
                                                                EnhancedDoubleStatistics::combine)));
      EnhancedDoubleStatistics statistics = stats.value();
      double max = statistics.getMax();
      double min = statistics.getMin();
      double binSize = ((max - min) / bins.length);
      double sum = min;
      for (int i = 0; i < bins.length; i++) {
         sum += binSize;
         bins[i] = sum;
      }
   }

   @Override
   public void reset() {
      if (bins != null) {
         Arrays.fill(bins, 0);
      }
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.map(f -> {
         for (int i = 0; i < bins.length; i++) {
            if (f.getValue() < bins[i]) {
               return Feature.TRUE(f.getName(), Integer.toString(i));
            }
         }
         return Feature.TRUE(f.getName(), Integer.toString(bins.length - 1));
      });
   }


   @Override
   public String describe() {
      if (applyToAll()) {
         return "BinTransform{numberOfBins=" + bins.length + "}";
      }
      return "BinTransform[" + getRestriction() + "]{numberOfBins=" + bins.length + "}";
   }

   @Override
   public void write(@NonNull StructuredWriter writer) throws IOException {
      if (!applyToAll()) {
         writer.writeKeyValue("restriction", getRestriction());
      }
      writer.writeKeyValue("bins", new PrimitiveArrayList<>(bins, Double.class));
   }

   @Override
   public void read(@NonNull StructuredReader reader) throws IOException {
      reset();
      while (reader.peek() != ElementType.END_OBJECT) {
         switch (reader.peekName()) {
            case "restriction":
               setRestriction(reader.nextKeyValue().v2.asString());
               break;
            case "bins":
               this.bins = Stream.of(reader.nextArray()).mapToDouble(Val::asDoubleValue).toArray();
               break;
         }
      }
   }


}// END OF RealToDiscreteTransform
