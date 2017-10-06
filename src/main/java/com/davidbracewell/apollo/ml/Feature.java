package com.davidbracewell.apollo.ml;

import com.davidbracewell.Copyable;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.string.StringUtils;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
@EqualsAndHashCode
public final class Feature implements Serializable, Comparable<Feature>, Copyable<Feature> {
   private static final long serialVersionUID = 1L;
   @Getter
   private final String prefix;
   @Getter
   private final String predicate;
   @Getter
   private final String position;
   @Getter
   private final double value;
   @Getter
   private final String featureName;

   public Feature(String prefix, String predicate, String position, double value) {
      Preconditions.checkArgument(StringUtils.isNotNullOrBlank(predicate), "The predicate must not be null or blank.");
      this.prefix = prefix;
      this.predicate = predicate;
      this.position = position;
      this.value = value;
      StringBuilder builder = new StringBuilder();
      if (StringUtils.isNotNullOrBlank(prefix)) {
         builder.append(prefix);
      }
      if (position != null) {
         builder.append('[').append(position).append(']');
      }

      if (builder.length() > 0) {
         builder.append("=");
      }

      this.featureName = builder.append(predicate).toString();
   }

   public Feature(String predicate, double value) {
      this(StringUtils.EMPTY, predicate, null, value);
   }

   public static Feature TRUE(String predicate) {
      return new Feature(predicate, 1.0);
   }

   public static Feature TRUE(String prefix, String predicate) {
      return new Feature(prefix, predicate, null, 1.0);
   }


   public static Feature TRUE(String prefix, String predicate, String position) {
      return new Feature(prefix, predicate, position, 1.0);
   }

   public static boolean isFalse(double value) {
      return value == 0 || value == -1;
   }

   public static boolean isFalse(String value) {
      return value.toLowerCase().equals("false");
   }

   public static boolean isTrue(String value) {
      return value.toLowerCase().equals("true");
   }

   public static boolean isTrue(double value) {
      return value == 1;
   }

   public static Feature real(String predicate, double value) {
      return new Feature(predicate, value);
   }

   public static Feature real(String prefix, String predicate, String position, double value) {
      return new Feature(prefix, predicate, position, value);
   }

   @Override
   public int compareTo(@NonNull Feature o) {
      return getFeatureName().compareTo(o.getFeatureName());
   }

   @Override
   public Feature copy() {
      return new Feature(prefix, predicate, position, value);
   }

   @Override
   public String toString() {
      return "(" + getFeatureName() + ", " + value + ")";
   }

}// END OF Feature
