package com.gengoai.apollo.hash.signature;

import com.gengoai.Parameters;
import com.gengoai.conversion.Cast;
import com.gengoai.conversion.Convert;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * The type Signature parameters.
 *
 * @author David B. Bracewell
 */
public class SignatureParameters implements Parameters, Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The constant Dimension.
    */
   public static final String DIMENSION = "DIMENSION";
   /**
    * The constant SignatureSize.
    */
   public static final String SIGNATURE_SIZE = "SIGNATURE_SIZE";
   /**
    * The constant MaxW.
    */
   public static final String MAX_W = "MAX_W";
   /**
    * The constant Threshold.
    */
   public static final String THRESHOLD = "THRESHOLD";

   private Integer dimension = null;
   private Integer signatureSize = null;
   private Integer maxW = null;
   private double threshold = 0.5;

   @Override
   public boolean contains(String name) {
      String upper = name.toUpperCase();
      return (upper.equals(DIMENSION) && dimension != null)
                || (upper.equals(SIGNATURE_SIZE) && signatureSize != null)
                || (upper.equals(MAX_W) && maxW != null)
                || (upper.equals(THRESHOLD));
   }

   @Override
   public SignatureParameters copy() {
      SignatureParameters toReturn = new SignatureParameters();
      getParameters().forEach(toReturn::set);
      return toReturn;
   }

   @Override
   public Object get(String name) {
      switch (name.toUpperCase()) {
         case DIMENSION:
            return dimension;
         case SIGNATURE_SIZE:
            return signatureSize;
         case MAX_W:
            return maxW;
         case THRESHOLD:
            return threshold;
         default:
            return null;
      }
   }

   @Override
   public <E> E getOrDefault(String name, E defaultValue) {
      return contains(name) ? getAs(name) : defaultValue;
   }

   @Override
   public Map<String, Object> getParameters() {
      Map<String, Object> toReturn = new HashMap<>();
      if (dimension != null) {
         toReturn.put(DIMENSION, dimension);
      }
      if (signatureSize != null) {
         toReturn.put(SIGNATURE_SIZE, signatureSize);
      }
      if (maxW != null) {
         toReturn.put(MAX_W, maxW);
      }
      toReturn.put(THRESHOLD, threshold);
      return toReturn;
   }


   @Override
   public SignatureParameters set(String name, Object value) {
      switch (name.toUpperCase()) {
         case DIMENSION:
            dimension = Cast.as(value, Integer.class);
            break;
         case SIGNATURE_SIZE:
            signatureSize = Cast.as(value, Integer.class);
            break;
         case MAX_W:
            maxW = Cast.as(value, Integer.class);
            break;
         case THRESHOLD:
            threshold = Convert.convert(value, double.class);
            break;
      }
      return this;
   }

   /**
    * Sets dimension.
    *
    * @param dimension the dimension
    * @return the dimension
    */
   public SignatureParameters setDimension(int dimension) {
      return set(DIMENSION, dimension);
   }

   /**
    * Sets signature size.
    *
    * @param signatureSize the signature size
    * @return the signature size
    */
   public SignatureParameters setSignatureSize(int signatureSize) {
      return set(SIGNATURE_SIZE, signatureSize);
   }

   /**
    * Gets dimension.
    *
    * @return the dimension
    */
   public int getDimension() {
      return dimension;
   }

   /**
    * Gets signature size.
    *
    * @return the signature size
    */
   public int getSignatureSize() {
      return signatureSize;
   }

   /**
    * Sets max w.
    *
    * @param maxW the max w
    * @return the max w
    */
   public SignatureParameters setMaxW(int maxW) {
      return set(MAX_W, maxW);
   }

   /**
    * Sets threshold.
    *
    * @param threshold the threshold
    * @return the threshold
    */
   public SignatureParameters setThreshold(double threshold) {
      return set(THRESHOLD, threshold);
   }

   /**
    * Gets max w.
    *
    * @return the max w
    */
   public int getMaxW() {
      return maxW == null ? 100 : maxW;
   }

   /**
    * Gets threshold.
    *
    * @return the threshold
    */
   public double getThreshold() {
      return threshold;
   }

}//END OF Parameters
