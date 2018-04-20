package com.gengoai.apollo.ml.encoder;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.guava.common.base.Preconditions;
import com.gengoai.stream.MStream;
import com.gengoai.apollo.ml.data.Dataset;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * <p>An encoder that hashes features names allowing for the feature space to be reduced to a predetermined number of
 * features.</p>
 *
 * @author David B. Bracewell
 */
public class HashingEncoder implements Encoder, Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   private final int numberOfFeatures;
   @Getter
   private final boolean binary;

   /**
    * Instantiates a new Hashing encoder with a default of 3000 features.
    */
   public HashingEncoder() {
      this(3000);
   }

   /**
    * Instantiates a new Hashing encoder.
    *
    * @param numberOfFeatures the number of features
    */
   public HashingEncoder(int numberOfFeatures) {
      this(numberOfFeatures, false);
   }

   public HashingEncoder(int numberOfFeatures, boolean binary) {
      Preconditions.checkArgument(numberOfFeatures > 0, "Must allow at least one feature.");
      this.numberOfFeatures = numberOfFeatures;
      this.binary = binary;
   }

   @Override
   public Encoder createNew() {
      return new HashingEncoder(numberOfFeatures, binary);
   }

   @Override
   public Object decode(double value) {
      return null;
   }

   @Override
   public double encode(@NonNull Object object) {
      return (object.hashCode() & 0x7fffffff) % numberOfFeatures;
   }

   @Override
   public void fit(Dataset<? extends Example> dataset) {

   }

   @Override
   public void fit(MStream<String> stream) {

   }

   @Override
   public void freeze() {

   }

   @Override
   public double get(Object object) {
      return encode(object);
   }

   @Override
   public boolean isFrozen() {
      return true;
   }

   @Override
   public int size() {
      return numberOfFeatures;
   }

   @Override
   public void unFreeze() {

   }

   @Override
   public List<Object> values() {
      return Collections.emptyList();
   }

}// END OF HashingEncoder
