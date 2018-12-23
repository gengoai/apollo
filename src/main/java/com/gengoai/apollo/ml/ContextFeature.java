package com.gengoai.apollo.ml;

import com.gengoai.string.Strings;

import java.io.Serializable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author David B. Bracewell
 */
public class ContextFeature implements Serializable {
   private static final Pattern featurePattern = Pattern.compile("^(.+?)\\[([-+]?\\d+)\\]$");
   private final String[] prefix;
   private final int[] offsets;
   private final String featurePrefix;


   public ContextFeature(String... patterns) {
      prefix = new String[patterns.length];
      offsets = new int[patterns.length];
      featurePrefix = Strings.join(patterns, "_");
      for (int i = 0; i < patterns.length; i++) {
         Matcher m = featurePattern.matcher(patterns[i]);
         if (m.find()) {
            prefix[i] = m.group(1) + "=";
            offsets[i] = Integer.parseInt(m.group(2));
         } else {
            throw new IllegalArgumentException(patterns[i] + " is not a legal pattern");
         }
      }
   }

   public Example apply(Example sequence) {
      for (ContextualIterator itr = sequence.iterator(); itr.hasNext(); ) {
         Example instance = itr.next();

         StringBuilder fName = new StringBuilder(featurePrefix).append("=");
         for (int i = 0; i < prefix.length; i++) {
            Feature f = itr.getContext(offsets[i]).getFeatureByPrefix(prefix[i]);
            if (i > 0) {
               fName.append("|");
            }
            fName.append(f.getSuffix());
         }
         instance.getFeatures().add(Feature.booleanFeature(fName.toString()));
      }
      return sequence;
   }


}//END OF ContextFeatures
