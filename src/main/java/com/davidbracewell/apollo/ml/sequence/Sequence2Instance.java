package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * The type Seq 2 inst.
 *
 * @author David B. Bracewell
 */
public class Sequence2Instance implements SerializableFunction<Sequence, Instance> {

   private final int maxSequenceSize;
   private final int labelIndex;


   public Sequence2Instance(int maxSequenceSize) {
      this(0, maxSequenceSize);
   }


   /**
    * Instantiates a new Seq 2 inst.
    *
    * @param maxSequenceSize the max sequence size
    */
   public Sequence2Instance(int labelIndex, int maxSequenceSize) {
      Preconditions.checkArgument(maxSequenceSize > 0, "Maximum sequence size must be > 0");
      this.maxSequenceSize = maxSequenceSize;
      this.labelIndex = labelIndex;
   }

   @Override
   public Instance apply(@NonNull Sequence sequence) {
      if (sequence.size() == 0) {
         return new Instance();
      }
      List<Feature> instFeatures = new ArrayList<>();
      int index = 0;
      for (Instance instance : sequence) {
         if (index >= maxSequenceSize) {
            break;
         }
         for (Feature feature : instance) {
            instFeatures.add(Feature.real(feature.getName() + "-" + index, feature.getValue()));
         }
         index++;
      }
      return Instance.create(instFeatures, sequence.get(labelIndex).getLabel());
   }

}// END OF Seq2Inst
