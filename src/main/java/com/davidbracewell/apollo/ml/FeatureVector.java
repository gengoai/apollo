package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;
import lombok.val;

import java.util.Map;
import java.util.stream.Stream;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * <p>A specialized sparse vector that stores an encoded label and an optionally an encoded predicted label. The
 * dimension of the vector is dynamic and depends on the underlying feature encoder.</p>
 *
 * @author David B. Bracewell
 */
public class FeatureVector extends SparseVector {
   private static final long serialVersionUID = 1L;
   private final EncoderPair encoderPair;

   /**
    * Instantiates a new Feature vector.
    *
    * @param encoderPair the feature encoder
    */
   public FeatureVector(@NonNull EncoderPair encoderPair) {
      super(0);
      this.encoderPair = encoderPair;
   }

   public Stream<Map.Entry<Object, Double>> decodedFeatureStream() {
      final Encoder encoder = getEncoderPair().getFeatureEncoder();
      if (size() == 0) {
         return Stream.empty();
      }
      return Streams.asStream(orderedNonZeroIterator())
                    .map(e -> $(encoder.decode(e.getIndex()), e.getValue()));
   }

   @Override
   public int dimension() {
      return encoderPair.getFeatureEncoder().size();
   }

   /**
    * Decodes the label returning its string form
    *
    * @return the decoded label
    */
   public String getDecodedLabel() {
      return Cast.as(encoderPair.decodeLabel(getLabel()));
   }

   /**
    * Gets the encoder pair used by the feature vector.
    *
    * @return the encoder pair
    */
   public EncoderPair getEncoderPair() {
      return encoderPair;
   }

   /**
    * Checks if the vector has a label assigned to it
    *
    * @return True if the vector has a label, False otherwise
    */
   public boolean hasLabel() {
      return Double.isFinite(getLabelAsDouble());
   }

   /**
    * Sets the given feature to the given value.
    *
    * @param featureName  the feature name
    * @param featureValue the feature value
    * @return True if the feature name was valid and the value was set
    */
   public boolean set(String featureName, double featureValue) {
      int index = (int) encoderPair.encodeFeature(featureName);
      if (index < 0) {
         return false;
      }
      set(index, featureValue);
      return true;
   }

   /**
    * Sets the value of the given feature.
    *
    * @param feature the feature
    * @return True if the feature is valid and its value set
    */
   public boolean set(Feature feature) {
      return feature != null && set(feature.getName(), feature.getValue());
   }

   /**
    * Sets the label of the vector.
    *
    * @param label the label
    */
   public Vector setLabel(Object label) {
      if (label instanceof Number) {
         super.setLabel(label);
      } else {
         double val = encoderPair.encodeLabel(label);
         if (val == -1) {
            setLabel(null);
         } else {
            setLabel(val);
         }
      }
      return this;
   }

   /**
    * Sets the label of the vector.
    *
    * @param label the label
    */
   public Vector setLabel(double label) {
      super.setLabel(label);
      return this;
   }

   /**
    * Sets the predicted label of the vector.
    *
    * @param label the predicted label
    */
   public Vector setPredicted(Object label) {
      val encodeLabel = encoderPair.encodeLabel(label);
      if (encodeLabel == -1) {
         setPredicted(Double.NaN);
      } else {
         setPredicted(encodeLabel);
      }
      return this;
   }

   /**
    * Transforms the feature vector using a new encoder pair.
    *
    * @param newEncoderPair the new feature encoder
    * @return the feature vector
    */
   public FeatureVector transform(@NonNull EncoderPair newEncoderPair) {
      FeatureVector newVector = new FeatureVector(newEncoderPair);
      forEachSparse(entry -> newVector.set(encoderPair.decodeFeature(entry.getIndex()).toString(),
                                           entry.getValue()
                                          )
                   );
      newVector.setLabel(getDecodedLabel());
      return newVector;
   }

}// END OF FeatureVector
