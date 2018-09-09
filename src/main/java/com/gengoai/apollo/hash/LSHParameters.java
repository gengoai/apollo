package com.gengoai.apollo.hash;

import com.gengoai.apollo.hash.signature.SignatureParameters;
import com.gengoai.conversion.Cast;
import com.gengoai.conversion.Convert;

import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class LSHParameters extends SignatureParameters {
   public static final String BANDS = "BANDS";
   public static final String BUCKETS = "BUCKETS";
   public static final String SIGNATURE_FUNCTION = "SIGNATURE_FUNCTION";
   private static final long serialVersionUID = 1L;
   private int bands = 5;
   private int buckets = 20;
   private String signatureFunction = "COSINE";

   @Override
   public boolean contains(String name) {
      String upper = name.toUpperCase();
      return super.contains(name)
                || (upper.equals(SIGNATURE_FUNCTION) && signatureFunction != null)
                || (upper.equals(BANDS))
                || (upper.equals(BUCKETS));
   }

   @Override
   public LSHParameters copy() {
      LSHParameters toReturn = new LSHParameters();
      getParameters().forEach(toReturn::set);
      return toReturn;
   }

   @Override
   public Object get(String name) {
      String upper = name.toUpperCase();
      switch (upper) {
         case BANDS:
            return bands;
         case BUCKETS:
            return buckets;
         case SIGNATURE_FUNCTION:
            return signatureFunction;
         default:
            return super.get(name);
      }
   }

   public int getBands() {
      return bands;
   }

   public LSHParameters setBands(Integer bands) {
      this.bands = bands;
      return this;
   }

   public int getBuckets() {
      return buckets;
   }

   public LSHParameters setBuckets(Integer buckets) {
      this.buckets = buckets;
      return this;
   }

   @Override
   public Map<String, Object> getParameters() {
      Map<String, Object> toReturn = super.getParameters();
      toReturn.put(BANDS, bands);
      toReturn.put(BUCKETS, buckets);
      if (signatureFunction != null) {
         toReturn.put(SIGNATURE_FUNCTION, signatureFunction);
      }
      return toReturn;
   }

   public String getSignatureFunction() {
      return signatureFunction;
   }

   public LSHParameters setSignatureFunction(String signatureFunction) {
      this.signatureFunction = signatureFunction;
      return this;
   }

   @Override
   public LSHParameters set(String name, Object value) {
      String upper = name.toUpperCase();
      switch (upper) {
         case BANDS:
            bands = Convert.convert(value, int.class);
            break;
         case BUCKETS:
            buckets = Convert.convert(value, int.class);
            break;
         case SIGNATURE_FUNCTION:
            signatureFunction = Cast.as(value, String.class);
            break;
         default:
            super.set(name, value);
      }
      return this;
   }


}//END OF LSHParameters
