package com.davidbracewell.apollo.ml;

import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredSerializable;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.reflection.Reflect;
import com.davidbracewell.reflection.ReflectionException;
import lombok.EqualsAndHashCode;

import java.io.IOException;
import java.io.Serializable;

/**
 * The type Encoder pair.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = false)
public final class EncoderPair implements Serializable, StructuredSerializable {
   private static final long serialVersionUID = 1L;
   private final LabelEncoder labelEncoder;
   private final Encoder featureEncoder;

   /**
    * Instantiates a new Encoder pair.
    *
    * @param labelEncoder   the label encoder
    * @param featureEncoder the feature encoder
    */
   public EncoderPair(LabelEncoder labelEncoder, Encoder featureEncoder) {
      this.labelEncoder = labelEncoder;
      this.featureEncoder = featureEncoder;
   }

   protected EncoderPair() {
      this.labelEncoder = null;
      this.featureEncoder = null;
   }

   private Object createEncoder(StructuredReader reader) throws IOException {
      reader.beginObject();
      Class<?> clazz = reader.nextKeyValue("class").asClass();
      Object o = reader.nextKeyValue("encoder", clazz);
      reader.endObject();
      return o;
   }

   /**
    * Create new encoder pair.
    *
    * @return the encoder pair
    */
   public EncoderPair createNew() {
      return new EncoderPair(labelEncoder.createNew(), featureEncoder.createNew());
   }

   /**
    * Decode feature object.
    *
    * @param value the value
    * @return the object
    */
   public Object decodeFeature(double value) {
      return featureEncoder.decode(value);
   }

   /**
    * Decode label object.
    *
    * @param value the value
    * @return the object
    */
   public Object decodeLabel(double value) {
      return labelEncoder.decode(value);
   }

   /**
    * Encode t.
    *
    * @param <T>     the type parameter
    * @param example the example
    * @return the t
    */
   public <T extends Example> T encode(T example) {
      if (example != null) {
         labelEncoder.encode(example.getLabelSpace());
         featureEncoder.encode(example.getFeatureSpace());
      }
      return example;
   }

   /**
    * Encode feature double.
    *
    * @param feature the feature
    * @return the double
    */
   public double encodeFeature(Object feature) {
      return featureEncoder.encode(feature);
   }

   /**
    * Encode label double.
    *
    * @param label the label
    * @return the double
    */
   public double encodeLabel(Object label) {
      return labelEncoder.encode(label);
   }

   /**
    * Gets the index of the given feature
    *
    * @param featureName the feature to lookup
    * @return the index (int value) of the encoded feature
    */
   public int featureIndex(Object featureName) {
      return featureEncoder.index(featureName);
   }

   /**
    * Freeze.
    */
   public void freeze() {
      this.labelEncoder.freeze();
      this.featureEncoder.freeze();
   }

   /**
    * Gets feature encoder.
    *
    * @return the feature encoder
    */
   public Encoder getFeatureEncoder() {
      return featureEncoder;
   }

   /**
    * Gets label encoder.
    *
    * @return the label encoder
    */
   public LabelEncoder getLabelEncoder() {
      return labelEncoder;
   }

   /**
    * Gets the index of the given label
    *
    * @param label the label to lookup
    * @return the index (int value) of the encoded label
    */
   public int labelIndex(Object label) {
      return labelEncoder.index(label);
   }

   /**
    * Number of features int.
    *
    * @return the int
    */
   public int numberOfFeatures() {
      return featureEncoder.size();
   }

   /**
    * Number of labels int.
    *
    * @return the int
    */
   public int numberOfLabels() {
      return labelEncoder.size();
   }

   @Override
   public void read(StructuredReader reader) throws IOException {
      try {
         while (reader.peek() != ElementType.END_OBJECT) {
            switch (reader.peekName()) {
               case "features":
                  Reflect.onObject(this).allowPrivilegedAccess().set("featureEncoder", createEncoder(reader));
                  break;
               case "labels":
                  Reflect.onObject(this).allowPrivilegedAccess().set("labelEncoder", createEncoder(reader));
                  break;
               default:
                  throw new IOException("Unexpected " + reader.peekName() + " [" + reader.peek() + "]");
            }
         }
      } catch (ReflectionException e) {
         throw new IOException(e);
      }
   }

   @Override
   public void write(StructuredWriter writer) throws IOException {
      writer
         .beginObject("features")
         .writeKeyValue("class", featureEncoder.getClass().getName())
         .writeKeyValue("encoder", featureEncoder)
         .endObject()
         .beginObject("labels")
         .writeKeyValue("class", labelEncoder.getClass().getName())
         .writeKeyValue("encoder", labelEncoder)
         .endObject();
   }

}// END OF EncoderPair
