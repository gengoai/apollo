package com.gengoai.apollo.ml;

import com.gengoai.string.Strings;

import java.io.Serializable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * <P>Creates features for sequences based on the surrounding context of a given example. Patterns are defined as
 * follows:</P>
 * <code>
 * PREFIX[relative_offset]
 * </code>
 * <p>
 * where relative offset is an integer with a negative representing to the left and optional positive sign representing
 * to the right. Generated features are in the following form:
 * </p>
 * <code>
 * PREFIX1[offset]|PREFIX2[offset]=VALUE1|VALUE2
 * </code>
 *
 * @author David B. Bracewell
 */
public class ContextFeature implements Serializable {
   private static final Pattern featurePattern = Pattern.compile("^(.+?)\\[([-+]?\\d+)\\]$");
   private final String[] prefix;
   private final int[] offsets;
   private final String featurePrefix;
   private final boolean ignoreEmptyContext;


   /**
    * Convenience method for creating a ContextFeature from a given pattern. Will use beginning and end of sequence
    * markers as needed.
    *
    * @param pattern the pattern
    * @return the context feature
    */
   public static ContextFeature contextFeature(String pattern) {
      return new ContextFeature(pattern);
   }

   /**
    * Convenience method for creating a ContextFeature from a given pattern. Will use or ignore beginning and end of
    * sequence markers as directed by the given <code>ignoreEmptyContext</code> parameters.
    *
    * @param ignoreEmptyContext the ignore empty context
    * @param pattern            the pattern
    * @return the context feature
    */
   public static ContextFeature contextFeature(boolean ignoreEmptyContext, String pattern) {
      return new ContextFeature(ignoreEmptyContext, pattern);
   }

   /**
    * Instantiates a new Context feature.Will use beginning and end of sequence markers as needed.
    *
    * @param pattern the pattern
    */
   public ContextFeature(String pattern) {
      this(false, pattern);
   }

   /**
    * Instantiates a new Context feature.
    *
    * @param ignoreEmptyContext True - ignore features where the expansion goes beyond the beginning or end of
    *                           sequence.
    * @param pattern            the pattern
    */
   public ContextFeature(boolean ignoreEmptyContext, String pattern) {
      String[] patterns = pattern.split("\\|");
      this.ignoreEmptyContext = ignoreEmptyContext;
      prefix = new String[patterns.length];
      offsets = new int[patterns.length];
      featurePrefix = Strings.join(patterns, "|");
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

   /**
    * Applies the extractor to the given example.
    *
    * @param sequence the sequence to create contextual features for
    * @return the example with contextual features
    */
   public Example apply(Example sequence) {
      for (ContextualIterator itr = sequence.iterator(); itr.hasNext(); ) {
         Example instance = itr.next();

         StringBuilder fName = new StringBuilder(featurePrefix).append("=");
         for (int i = 0; i < prefix.length; i++) {
            Feature f = itr.getContext(offsets[i]).getFeatureByPrefix(prefix[i]);
            if (ignoreEmptyContext) {
               if (f.getSuffix().startsWith("__BOS-")
                      || f.getSuffix().startsWith("__EOS-")) {
                  fName = null;
                  break;
               }
            }
            if (i > 0) {
               fName.append("|");
            }
            fName.append(f.getSuffix());
         }
         if (fName != null) {
            instance.getFeatures().add(Feature.booleanFeature(fName.toString()));
         }
      }
      return sequence;
   }


}//END OF ContextFeature
