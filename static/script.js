document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const urlForm = document.getElementById("prediction-form");
  const urlInput = document.querySelector('.url-input');
  const heroVerifyButton = document.querySelector('.hero-buttons .verify-button');
  const heroLearnMoreButton = document.querySelector('.hero-buttons .learn-more-button');
  const urlInputSection = document.querySelector('.url-input-section');

  // Smooth scroll function
  const smoothScroll = (element) => {
    element.scrollIntoView({
      behavior: 'smooth',
      block: 'center'
    });
  };

  // Add pulse animation to URL input
  const addPulseToInput = () => {
    urlInput.classList.add('pulse-animation');
    setTimeout(() => {
      urlInput.classList.remove('pulse-animation');
    }, 1000);
  };

  // Hero Verify button click handler
  heroVerifyButton.addEventListener('click', () => {
    smoothScroll(urlInputSection);
    addPulseToInput();
    urlInput.focus();
  });

  // Learn More button click handler
  heroLearnMoreButton.addEventListener('click', () => {
    const featuresSection = document.querySelector('.features-section');
    if (featuresSection) {
      smoothScroll(featuresSection);
    }
  });

  // State
  let isLoading = false;
  let currentAnalysis = null;

  // URL validation
  function isValidUrl(url) {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  // Show/hide loading state
  function setLoading(loading) {
    isLoading = loading;
    loadingSpinner.classList.toggle("hidden", !loading);
    urlInput.disabled = loading;
    verifyButton.disabled = loading;
  }

  // Show message
  function showMessage(element, message) {
    element.textContent = message;
    element.classList.remove("hidden");
    element.classList.add("message", "slide-in");
    setTimeout(() => {
      element.classList.add("hidden");
      element.classList.remove("message", "slide-in");
    }, 5000);
  }

  // Mock API call - Replace with actual API integration
  async function analyzeUrl(url) {
    setLoading(true);

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Mock response - Replace with actual API call
      const response = {
        isReal: Math.random() > 0.5,
        confidence: Math.floor(Math.random() * 30) + 70,
        analysis: {
          source: "Verified",
          date: new Date().toISOString(),
          category: "News",
        },
      };

      setLoading(false);
      return response;
    } catch (error) {
      setLoading(false);
      throw new Error("Failed to analyze URL");
    }
  }

  // Form submission
  urlForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const url = urlInput.value.trim();

    if (!isValidUrl(url)) {
      showMessage(errorMessage, "Please enter a valid URL");
      return;
    }

    try {
      const result = await analyzeUrl(url);
      currentAnalysis = result;

      // Update UI based on result
      if (result.isReal) {
        realButton.classList.add("bg-blue-700");
        fakeButton.classList.remove("bg-red-700");
        showMessage(
          successMessage,
          `Article appears to be real (${result.confidence}% confidence)`,
        );
      } else {
        fakeButton.classList.add("bg-red-700");
        realButton.classList.remove("bg-blue-700");
        showMessage(
          errorMessage,
          `Article appears to be fake (${result.confidence}% confidence)`,
        );
      }
    } catch (error) {
      showMessage(errorMessage, error.message);
    }
  });

  // Button interactions
  document.querySelectorAll("button").forEach((button) => {
    button.addEventListener("mouseenter", () => {
      button.classList.add("button-hover");
    });

    button.addEventListener("mouseleave", () => {
      button.classList.remove("button-hover");
    });
  });

  // Smooth scroll for navigation links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      e.preventDefault();
      const element = document.querySelector(anchor.getAttribute("href"));
      if (element) {
        element.scrollIntoView({
          behavior: "smooth",
        });
      }
    });
  });

  // Error boundary
  window.addEventListener("error", (event) => {
    console.error("Application error:", event.error);
    showMessage(
      errorMessage,
      "An unexpected error occurred. Please try again.",
    );
  });

  // Initialize tooltips
  const tooltipElements = document.querySelectorAll("[data-tooltip]");
  tooltipElements.forEach((element) => {
    element.addEventListener("mouseenter", (e) => {
      const tooltip = document.createElement("div");
      tooltip.className = "tooltip fade-in";
      tooltip.textContent = element.dataset.tooltip;
      document.body.appendChild(tooltip);

      const rect = element.getBoundingClientRect();
      tooltip.style.top = `${rect.top - tooltip.offsetHeight - 10}px`;
      tooltip.style.left = `${rect.left + (rect.width - tooltip.offsetWidth) / 2}px`;
    });

    element.addEventListener("mouseleave", () => {
      const tooltip = document.querySelector(".tooltip");
      if (tooltip) {
        tooltip.remove();
      }
    });
  });

  // Handle hero buttons
  const heroButtons = document.querySelector(".hero-buttons");
  if (heroButtons) {
    const verifyButton = heroButtons.querySelector(".verify-button");
    if (verifyButton) {
      verifyButton.addEventListener("click", () => {
        const urlSection = document.querySelector(".url-input-section");
        if (urlSection) {
          urlSection.scrollIntoView({ behavior: "smooth" });
          urlInput.focus();
        }
      });
    }
  }

  // Login Modal Elements
  const loginButton = document.querySelector('.login-button');
  const loginModal = document.getElementById('loginModal');
  const signupModal = document.getElementById('signupModal');
  const closeButtons = document.querySelectorAll('.close-modal');
  const showSignupLink = document.getElementById('showSignup');
  const showLoginLink = document.getElementById('showLogin');
  const loginForm = document.getElementById('loginForm');
  const signupForm = document.getElementById('signupForm');

  // Modal Functions
  function openModal(modal) {
    modal.classList.add('show');
    document.body.style.overflow = 'hidden';
  }

  function closeModal(modal) {
    modal.classList.remove('show');
    document.body.style.overflow = '';
  }

  function closeAllModals() {
    [loginModal, signupModal].forEach(modal => closeModal(modal));
  }

  // Event Listeners
  loginButton.addEventListener('click', () => openModal(loginModal));

  closeButtons.forEach(button => {
    button.addEventListener('click', () => {
      closeAllModals();
    });
  });

  showSignupLink.addEventListener('click', (e) => {
    e.preventDefault();
    closeModal(loginModal);
    openModal(signupModal);
  });

  showLoginLink.addEventListener('click', (e) => {
    e.preventDefault();
    closeModal(signupModal);
    openModal(loginModal);
  });

  // Close modal when clicking outside
  window.addEventListener('click', (e) => {
    if (e.target === loginModal || e.target === signupModal) {
      closeAllModals();
    }
  });

  // Handle Google Sign-in Response
  window.handleCredentialResponse = async (response) => {
    try {
      // Here you would verify the credential with your backend
      const credential = response.credential;
      
      // For demo purposes, just log the user in
      const userData = JSON.parse(atob(credential.split('.')[1]));
      console.log('Logged in user:', userData.name);
      
      // Update UI for logged-in state
      loginButton.textContent = userData.name;
      closeAllModals();
      
      // You can add more UI updates here
      
    } catch (error) {
      console.error('Error processing Google sign-in:', error);
      alert('Failed to process Google sign-in. Please try again.');
    }
  };

  // Form Submissions
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    try {
      // Here you would make an API call to your backend
      console.log('Login attempt:', { email });
      
      // For demo purposes
      alert('Login successful!');
      closeAllModals();
      loginButton.textContent = email.split('@')[0];
      
    } catch (error) {
      console.error('Login error:', error);
      alert('Login failed. Please try again.');
    }
  });

  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fullName = document.getElementById('fullName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;

    try {
      // Here you would make an API call to your backend
      console.log('Signup attempt:', { fullName, email });
      
      // For demo purposes
      alert('Account created successfully! Please log in.');
      closeModal(signupModal);
      openModal(loginModal);
      
    } catch (error) {
      console.error('Signup error:', error);
      alert('Failed to create account. Please try again.');
    }
  });
});
