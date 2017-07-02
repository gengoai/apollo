package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Instance;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public interface TransitionFeature {

   Iterator<String> extract(final Context<Instance> iterator);

}//END OF TransitionFeature
