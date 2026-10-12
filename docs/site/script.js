/* Forge — Site Interactions */

(function () {
  'use strict';

  // ── Scroll-triggered fade-in via IntersectionObserver ──
  var fadeElements = document.querySelectorAll('.fade-in');

  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 }
    );

    fadeElements.forEach(function (el) {
      observer.observe(el);
    });
  } else {
    // Fallback: reveal everything immediately
    fadeElements.forEach(function (el) {
      el.classList.add('revealed');
    });
  }

  // ── Mobile nav toggle ──
  var hamburger = document.querySelector('.hamburger');
  var nav = document.querySelector('.nav');
  var navLinks = document.querySelectorAll('.nav-links a');

  if (hamburger && nav) {
    hamburger.addEventListener('click', function () {
      nav.classList.toggle('nav-open');
    });

    // Close mobile menu when a nav link is clicked
    navLinks.forEach(function (link) {
      link.addEventListener('click', function () {
        nav.classList.remove('nav-open');
      });
    });
  }
})();
